import time
from typing import Dict, Any, List, Optional
import numpy as np
import plotly.graph_objects as go
from ..core.adapter import FrameworkAdapter

class PruningAnimator:
    """The 'X-Ray' Visualizer: interactive animations of the pruning process.
    
    This module generates 4-stage animations to help researchers and 
    stakeholders understand how ReduCNN identifies dependencies and 
    physically shrinks the model.
    """
    
    def __init__(self, adapter: FrameworkAdapter):
        """Initializes the animator with a framework adapter.
        
        Args:
            adapter (FrameworkAdapter): The backend-specific adapter.
        """
        self.adapter = adapter

    def render(self, fig: go.Figure, renderer: Optional[str] = None) -> go.Figure:
        """Renders a Plotly figure with notebook-safe fallbacks.

        This helps in environments where the default Plotly renderer is not
        configured and `fig.show()` would otherwise produce an empty output.
        """
        if fig is None:
            return go.Figure()
        try:
            import plotly.io as pio

            if renderer:
                fig.show(renderer=renderer)
            else:
                default_renderer = str(pio.renderers.default or "").strip()
                if not default_renderer:
                    pio.renderers.default = "notebook_connected"
                fig.show(renderer=str(pio.renderers.default))
            return fig
        except Exception:
            try:
                from IPython.display import HTML, display

                # Keep fallback self-contained to avoid CDN/network failures.
                display(HTML(fig.to_html(include_plotlyjs=True, full_html=False)))
            except Exception as e:
                print(f"⚠️ Unable to render Plotly figure inline: {e}")
            return fig

    def export_html(self, fig: go.Figure, path: str = "pruning_xray.html") -> str:
        """Exports an animation figure to an embeddable HTML file."""
        # Use inline Plotly JS so exported files render without internet/CDN.
        fig.write_html(path, include_plotlyjs=True, full_html=True)
        return path

    def plot_architecture(self, model: Any, title: str = "Model Architecture", render: bool = True) -> go.Figure:
        """Generates a static, high-quality plot of the model's architecture graph.
        
        Useful for presentations and analyzing dependencies before pruning.
        """
        graph = self.adapter.trace_graph(model)
        nodes = graph.get("nodes", {})
        node_names = sorted(nodes.keys())
        if not node_names:
            return go.Figure()

        # 1) Layout with depth-aware placement
        depths = {name: 0 for name in node_names}
        queue = [n for n in node_names if not any(i in node_names for i in nodes[n].get("inputs", []))]
        visited = set()
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            curr_depth = depths[curr]
            for out in nodes[curr].get("outputs", []):
                if out in node_names:
                    depths[out] = max(depths.get(out, 0), curr_depth + 1)
                    queue.append(out)

        node_names = sorted(node_names, key=lambda n: (depths[n], n))
        depth_groups: Dict[int, List[str]] = {}
        for n in node_names:
            depth_groups.setdefault(depths[n], []).append(n)
            
        coords = {}
        for d, members in depth_groups.items():
            span = max(len(members) - 1, 1)
            for idx, n in enumerate(members):
                y = (idx - span / 2.0) * 1.5
                coords[n] = (float(d), float(y))
                
        node_x = [coords[n][0] for n in node_names]
        node_y = [coords[n][1] for n in node_names]
        name_to_idx = {n: i for i, n in enumerate(node_names)}

        # 2) Edges
        edge_x, edge_y = [], []
        for src in node_names:
            for dst in nodes[src].get("outputs", []):
                if dst in name_to_idx:
                    i, j = name_to_idx[src], name_to_idx[dst]
                    edge_x.extend([node_x[i], node_x[j], None])
                    edge_y.extend([node_y[i], node_y[j], None])
                    
        edges_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color="#888"),
            hoverinfo="none",
            mode="lines",
            showlegend=False
        )

        # 3) Nodes
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        node_colors = []
        hover_text = []
        for n in node_names:
            cid = nodes[n].get("cluster")
            node_colors.append(palette[cid % len(palette)] if cid is not None else "#aaaaaa")
            hover_text.append(
                f"<b>{n}</b><br>"
                f"Type: {nodes[n].get('type', 'unknown')}<br>"
                f"Cluster ID: {cid if cid is not None else 'None'}"
            )

        nodes_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_names,
            textposition="top center",
            marker=dict(
                size=22,
                color=node_colors,
                line=dict(width=2, color="#333"),
            ),
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False
        )

        fig = go.Figure(data=[edges_trace, nodes_trace],
                        layout=go.Layout(
                            title=dict(text=title, font=dict(size=16)),
                            showlegend=False,
                            hovermode="closest",
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor="white"
                        ))
        
        if render:
            self.render(fig)
        return fig

    def generate_xray_animation(self, model: Any, score_map: Dict[str, np.ndarray], 
                                masks: Dict[str, np.ndarray]) -> go.Figure:
        """Creates an interactive 4-stage Plotly animation of the pruning process."""
        graph = self.adapter.trace_graph(model)
        nodes = graph["nodes"]
        node_names = sorted(nodes.keys())
        if not node_names:
            return go.Figure()

        # 1) Layout with depth-aware placement.
        depths = {name: 0 for name in node_names}
        queue = [n for n in node_names if not any(i in node_names for i in nodes[n].get("inputs", []))]
        visited = set()
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            curr_depth = depths[curr]
            for out in nodes[curr].get("outputs", []):
                if out in node_names:
                    depths[out] = max(depths[out], curr_depth + 1)
                    queue.append(out)

        node_names = sorted(node_names, key=lambda n: (depths[n], n))
        depth_groups: Dict[int, List[str]] = {}
        for n in node_names:
            depth_groups.setdefault(depths[n], []).append(n)
        coords = {}
        for d, members in depth_groups.items():
            span = max(len(members) - 1, 1)
            for idx, n in enumerate(members):
                y = (idx - span / 2.0) * 1.5
                coords[n] = (float(d), float(y))
        node_x = [coords[n][0] for n in node_names]
        node_y = [coords[n][1] for n in node_names]
        name_to_idx = {n: i for i, n in enumerate(node_names)}

        # 2) Base edge layer.
        edge_x, edge_y = [], []
        for src in node_names:
            for dst in nodes[src].get("outputs", []):
                if dst in name_to_idx:
                    i, j = name_to_idx[src], name_to_idx[dst]
                    edge_x.extend([node_x[i], node_x[j], None])
                    edge_y.extend([node_y[i], node_y[j], None])
        base_edges = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#c7c7c7"),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )

        # 3) Diagnostics metadata.
        score_mean = {}
        if score_map:
            for n, s in score_map.items():
                if np.asarray(s).size:
                    score_mean[n] = float(np.asarray(s).reshape(-1).mean())
        if score_mean:
            s_min = min(score_mean.values())
            s_max = max(score_mean.values())
        else:
            s_min, s_max = 0.0, 1.0

        keep_ratio = {}
        for n in node_names:
            if n in masks and np.asarray(masks[n]).size:
                keep_ratio[n] = float(np.asarray(masks[n]).mean())
            else:
                keep_ratio[n] = 1.0

        cluster_ok = {n: True for n in node_names}
        for _, members in graph.get("clusters", {}).items():
            valid = [m for m in members if m in masks]
            if len(valid) < 2:
                continue
            ref = np.asarray(masks[valid[0]]).astype(bool)
            for m in valid[1:]:
                same = np.array_equal(ref, np.asarray(masks[m]).astype(bool))
                cluster_ok[m] = same
                cluster_ok[valid[0]] = cluster_ok[valid[0]] and same

        hover_text = []
        for n in node_names:
            cluster_id = nodes[n].get("cluster")
            sc = score_mean.get(n, float("nan"))
            k = keep_ratio.get(n, 1.0)
            hover_text.append(
                f"{n}<br>"
                f"type={nodes[n].get('type', 'unknown')}<br>"
                f"cluster={cluster_id if cluster_id is not None else '-'}<br>"
                f"mean_score={sc:.4f}<br>"
                f"keep_ratio={k:.2%}<br>"
                f"cluster_consistent={cluster_ok.get(n, True)}"
            )

        # 4) Build stages.
        cluster_colors = []
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
        for n in node_names:
            cid = nodes[n].get("cluster")
            cluster_colors.append(palette[cid % len(palette)] if cid is not None else "#7f7f7f")

        imp_colors = []
        for n in node_names:
            if n in score_mean:
                imp_colors.append((score_mean[n] - s_min) / (s_max - s_min + 1e-8))
            else:
                imp_colors.append(0.5)

        consistency_colors = []
        for n in node_names:
            cid = nodes[n].get("cluster")
            if cid is None:
                consistency_colors.append("#7f7f7f")
            else:
                consistency_colors.append("#2ca02c" if cluster_ok.get(n, True) else "#d62728")

        shrink_sizes = [10 + 24 * keep_ratio[n] for n in node_names]

        frames = [
            go.Frame(
                data=[
                    base_edges,
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode="markers+text",
                        text=node_names,
                        textposition="top center",
                        marker=dict(
                            size=18,
                            color=cluster_colors,
                            showscale=False,
                            line=dict(width=1, color="#333"),
                        ),
                        hovertext=hover_text,
                        hoverinfo="text",
                        showlegend=False,
                    ),
                ],
                name="Discovery",
            ),
            go.Frame(
                data=[
                    base_edges,
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode="markers+text",
                        text=node_names,
                        textposition="top center",
                        marker=dict(
                            color=imp_colors,
                            colorscale="Viridis",
                            size=18,
                            cmin=0.0,
                            cmax=1.0,
                            showscale=True,
                            colorbar=dict(title="Norm Score"),
                            line=dict(width=1, color="#333"),
                        ),
                        hovertext=hover_text,
                        hoverinfo="text",
                        showlegend=False,
                    ),
                ],
                name="Importance",
            ),
            go.Frame(
                data=[
                    base_edges,
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode="markers+text",
                        text=node_names,
                        textposition="top center",
                        marker=dict(
                            size=18,
                            color=consistency_colors,
                            showscale=False,
                            line=dict(width=1, color="#333"),
                        ),
                        hovertext=hover_text,
                        hoverinfo="text",
                        showlegend=False,
                    ),
                ],
                name="Consistency",
            ),
            go.Frame(
                data=[
                    base_edges,
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode="markers+text",
                        text=node_names,
                        textposition="top center",
                        marker=dict(
                            size=shrink_sizes,
                            color=consistency_colors,
                            showscale=False,
                            line=dict(width=1, color="#333"),
                        ),
                        hovertext=hover_text,
                        hoverinfo="text",
                        showlegend=False,
                    ),
                ],
                name="Shrink",
            ),
        ]

        # 5) Figure + controls.
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="ReduCNN X-Ray: Dependency, Importance, Consistency, and Shrinkage",
                template="plotly_white",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                annotations=[
                    dict(
                        x=0.01,
                        y=1.08,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=12, color="#333"),
                        text=(
                            "Discovery: cluster colors | Importance: colorbar score | "
                            "Consistency: green=matched red=mismatch gray=no-cluster | "
                            "Shrinkage: larger node=more channels kept"
                        ),
                    )
                ],
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(
                                label="Play All Stages",
                                method="animate",
                                args=[
                                    [f.name for f in frames],
                                    {"frame": {"duration": 500, "redraw": True}, "transition": {"duration": 250}},
                                ],
                            )
                        ],
                    )
                ],
            ),
            frames=frames,
        )
        fig.update_layout(
            sliders=[
                {
                    "active": 0,
                    "steps": [
                        {"label": "1. Discovery", "method": "animate", "args": [["Discovery"]]},
                        {"label": "2. Importance", "method": "animate", "args": [["Importance"]]},
                        {"label": "3. Consistency", "method": "animate", "args": [["Consistency"]]},
                        {"label": "4. Shrinkage", "method": "animate", "args": [["Shrink"]]},
                    ],
                }
            ]
        )
        return fig

    def generate_hybrid_heatmap(self, model: Any) -> go.Figure:
        """Visualizes the depth-based hybrid weights as a true heatmap."""
        graph = self.adapter.trace_graph(model)
        nodes = graph["nodes"]
        prunable_layers = [n for n, d in nodes.items() if d.get("type") == "conv2d"]
        n_layers = max(len(prunable_layers), 20)

        depths = np.linspace(0.0, 1.0, n_layers)
        w_l1, w_act, w_taylor = [], [], []

        from ..pruner.meta_criteria import HybridMetaPruner
        meta = HybridMetaPruner(self.adapter, mode='smooth')

        for d in depths:
            w1, w2, w3 = meta._get_weights(float(d))
            w_l1.append(float(w1))
            w_act.append(float(w2))
            w_taylor.append(float(w3))

        # Rows are metrics, columns are depth bins.
        z = np.vstack([w_l1, w_act, w_taylor])

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=z,
                    x=depths,
                    y=["L1-Norm (Early)", "Activations (Middle)", "Taylor (Deep)"],
                    colorscale="Viridis",
                    zmin=0.0,
                    zmax=1.0,
                    colorbar=dict(title="Weight"),
                    hovertemplate=(
                        "Metric: %{y}<br>"
                        "Depth: %{x:.2f}<br>"
                        "Weight: %{z:.3f}<extra></extra>"
                    ),
                )
            ]
        )

        fig.update_layout(
            title="ReduCNN v0.6.6 Hybrid Blending Heatmap",
            xaxis_title="Relative Network Depth (Input -> Output)",
            yaxis_title="Metric",
            template="plotly_white",
        )
        return fig

    def generate_hybrid_contribution_graph(self, model: Any, loader: Any, mode: str = "smooth") -> go.Figure:
        """Visualizes per-layer hybrid metric contributions on the traced dependency graph.

        Node color uses an RGB blend:
        - R: L1 contribution
        - G: Activation contribution
        - B: Taylor contribution
        """
        graph = self.adapter.trace_graph(model)
        nodes = graph.get("nodes", {})
        node_names = sorted(nodes.keys())
        if not node_names:
            return go.Figure()

        # Build a depth-aware layout.
        depths = {name: 0 for name in node_names}
        queue = [n for n in node_names if not any(i in node_names for i in nodes[n].get("inputs", []))]
        visited = set()
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            curr_depth = depths[curr]
            for out in nodes[curr].get("outputs", []):
                if out in node_names:
                    depths[out] = max(depths[out], curr_depth + 1)
                    queue.append(out)

        node_names = sorted(node_names, key=lambda n: (depths[n], n))
        depth_groups: Dict[int, List[str]] = {}
        for n in node_names:
            depth_groups.setdefault(depths[n], []).append(n)
        coords = {}
        for d, members in depth_groups.items():
            span = max(len(members) - 1, 1)
            for idx, n in enumerate(members):
                y = (idx - span / 2.0) * 1.5
                coords[n] = (float(d), float(y))
        node_x = [coords[n][0] for n in node_names]
        node_y = [coords[n][1] for n in node_names]
        name_to_idx = {n: i for i, n in enumerate(node_names)}

        edge_x, edge_y = [], []
        for src in node_names:
            for dst in nodes[src].get("outputs", []):
                if dst in name_to_idx:
                    i, j = name_to_idx[src], name_to_idx[dst]
                    edge_x.extend([node_x[i], node_x[j], None])
                    edge_y.extend([node_y[i], node_y[j], None])

        # Collect metric maps once and compute layer-level contribution shares.
        from ..pruner.meta_criteria import HybridMetaPruner

        meta = HybridMetaPruner(self.adapter, mode=mode)
        metrics = self.adapter.get_multi_metric_scores(model, loader, ["l1_norm", "mean_abs_act", "taylor"])
        prunable_layers = [n for n in node_names if nodes[n].get("type") == "conv2d"]

        contributions: Dict[str, Dict[str, float]] = {}
        for i, layer_name in enumerate(prunable_layers):
            depth = i / (len(prunable_layers) - 1) if len(prunable_layers) > 1 else 0.0
            w_l1, w_act, w_taylor = meta._get_weights(depth)

            s_l1 = meta._normalize(np.asarray(metrics["l1_norm"][layer_name]).reshape(-1))
            s_act = meta._normalize(np.asarray(metrics["mean_abs_act"][layer_name]).reshape(-1))
            s_taylor = meta._normalize(np.asarray(metrics["taylor"][layer_name]).reshape(-1))

            c_l1 = float((w_l1 * s_l1).mean())
            c_act = float((w_act * s_act).mean())
            c_taylor = float((w_taylor * s_taylor).mean())
            total = max(c_l1 + c_act + c_taylor, 1e-12)
            contributions[layer_name] = {
                "l1_norm": c_l1 / total,
                "mean_abs_act": c_act / total,
                "taylor": c_taylor / total,
                "w_l1": float(w_l1),
                "w_act": float(w_act),
                "w_taylor": float(w_taylor),
            }

        hover_text = []
        for n in node_names:
            if n in contributions:
                c = contributions[n]
                hover_text.append(
                    f"{n}<br>"
                    f"type={nodes[n].get('type', 'unknown')}<br>"
                    f"L1 share={c['l1_norm']:.2%} (w={c['w_l1']:.2f})<br>"
                    f"Act share={c['mean_abs_act']:.2%} (w={c['w_act']:.2f})<br>"
                    f"Taylor share={c['taylor']:.2%} (w={c['w_taylor']:.2f})"
                )
            else:
                hover_text.append(
                    f"{n}<br>"
                    f"type={nodes[n].get('type', 'unknown')}<br>"
                    "Not a prunable Conv2D node"
                )

        def node_colors(view_mode: str) -> List[str]:
            colors = []
            for n in node_names:
                if n not in contributions:
                    colors.append("#9e9e9e")
                    continue
                c = contributions[n]
                if view_mode == "L1":
                    r = int(40 + 215 * c["l1_norm"]); g = 40; b = 40
                elif view_mode == "Activation":
                    r = 40; g = int(40 + 215 * c["mean_abs_act"]); b = 40
                elif view_mode == "Taylor":
                    r = 40; g = 40; b = int(40 + 215 * c["taylor"])
                else:  # Blend
                    r = int(30 + 225 * c["l1_norm"])
                    g = int(30 + 225 * c["mean_abs_act"])
                    b = int(30 + 225 * c["taylor"])
                colors.append(f"rgb({r},{g},{b})")
            return colors

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#c7c7c7"),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )

        def node_trace(view_mode: str) -> go.Scatter:
            return go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=node_names,
                textposition="top center",
                marker=dict(size=20, color=node_colors(view_mode), line=dict(width=1, color="#333")),
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=False,
            )

        modes = ["L1", "Activation", "Taylor", "Blend"]
        frames = [go.Frame(data=[edge_trace, node_trace(m)], name=m) for m in modes]

        fig = go.Figure(
            data=frames[-1].data,
            frames=frames,
            layout=go.Layout(
                title="ReduCNN Hybrid Contribution Graph (Animated Views)",
                template="plotly_white",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(
                                label="Play Views",
                                method="animate",
                                args=[
                                    modes,
                                    {"frame": {"duration": 700, "redraw": True}, "transition": {"duration": 250}},
                                ],
                            )
                        ],
                    )
                ],
                annotations=[
                    dict(
                        x=0.01,
                        y=1.08,
                        xref="paper",
                        yref="paper",
                        text="Views: L1-only, Activation-only, Taylor-only, then RGB Blend",
                        showarrow=False,
                        font=dict(size=12, color="#333"),
                    )
                ],
            ),
        )
        fig.update_layout(
            sliders=[
                {
                    "active": len(modes) - 1,
                    "steps": [{"label": m, "method": "animate", "args": [[m]]} for m in modes],
                }
            ]
        )
        return fig

    def show_dependency_sweep(self, model: Any):
        """Visualizes the cluster discovery process."""
        print("🔍 Stage 1: Dependency Discovery (Scanning for clusters...)")
        graph = self.adapter.trace_graph(model)
        for c_id, members in graph["clusters"].items():
            print(f"Cluster {c_id}: {members}")

    def show_shrinkage(self, model: Any, masks: Dict[str, np.ndarray]):
        """Simulates the physical shrinking animation."""
        print("✂️ Stage 3: Physical Shrink (Simulating tensor reduction...)")
        for name, mask in masks.items():
            orig = len(mask)
            kept = np.sum(mask)
            print(f"Layer {name}: {orig} -> {kept} channels")
