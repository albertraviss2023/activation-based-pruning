import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..core.adapter import FrameworkAdapter
from .persistence import persist_plotly_figure

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
            try:
                persist_plotly_figure(fig, "plotly_render")
            except Exception:
                pass
            return fig
        except Exception:
            try:
                from IPython.display import HTML, display

                # Keep fallback self-contained to avoid CDN/network failures.
                display(HTML(fig.to_html(include_plotlyjs=True, full_html=False)))
                try:
                    persist_plotly_figure(fig, "plotly_render")
                except Exception:
                    pass
            except Exception as e:
                print(f"⚠️ Unable to render Plotly figure inline: {e}")
            return fig

    def export_html(self, fig: go.Figure, path: str = "pruning_xray.html") -> str:
        """Exports an animation figure to an embeddable HTML file."""
        # Use inline Plotly JS so exported files render without internet/CDN.
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out), include_plotlyjs=True, full_html=True)
        try:
            persist_plotly_figure(fig, out.stem)
        except Exception:
            pass
        return str(out)

    def _to_float_array(self, arr: Any) -> np.ndarray:
        return np.asarray(arr, dtype=np.float64).reshape(-1)

    def _normalize01(self, vals: np.ndarray) -> np.ndarray:
        v = self._to_float_array(vals)
        if v.size == 0:
            return v
        lo = float(np.min(v))
        hi = float(np.max(v))
        if hi <= lo + 1e-12:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    def _hex_to_rgb(self, color: str) -> Tuple[int, int, int]:
        c = color.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

    def _blend_hex(self, c1: str, c2: str, t: float) -> str:
        t = float(np.clip(t, 0.0, 1.0))
        r1, g1, b1 = self._hex_to_rgb(c1)
        r2, g2, b2 = self._hex_to_rgb(c2)
        r = int(round(r1 + (r2 - r1) * t))
        g = int(round(g1 + (g2 - g1) * t))
        b = int(round(b1 + (b2 - b1) * t))
        return f"rgb({r},{g},{b})"

    def _depth_layout(self, nodes: Dict[str, Dict[str, Any]]) -> Tuple[List[str], Dict[str, int], Dict[str, Tuple[float, float]]]:
        node_names = sorted(nodes.keys())
        if not node_names:
            return [], {}, {}

        depths = {name: 0 for name in node_names}
        queue = [n for n in node_names if not any(i in node_names for i in nodes[n].get("inputs", []))]
        if not queue:
            queue = list(node_names)
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

        ordered = sorted(node_names, key=lambda n: (depths[n], n))
        depth_groups: Dict[int, List[str]] = {}
        for n in ordered:
            depth_groups.setdefault(depths[n], []).append(n)

        coords: Dict[str, Tuple[float, float]] = {}
        for d, members in depth_groups.items():
            span = max(len(members) - 1, 1)
            for idx, n in enumerate(members):
                y = (idx - span / 2.0) * 1.5
                coords[n] = (float(d), float(y))
        return ordered, depths, coords

    def _short_name(self, name: str) -> str:
        if not name:
            return name
        if name in ("conv1", "fc"):
            return name
        parts = name.split(".")
        if len(parts) >= 4 and parts[0].startswith("layer") and parts[3] == "0" and parts[2] == "downsample":
            return f"{parts[0]}.{parts[1]}.ds0"
        if len(parts) >= 3 and parts[0].startswith("layer") and parts[2].startswith("conv"):
            return f"{parts[0]}.{parts[1]}.{parts[2]}"
        return name

    def _resolve_layer_channels(self, model: Any, node_names: List[str]) -> Dict[str, int]:
        """Best-effort channel/out-feature extraction for Torch and Keras models."""
        channels: Dict[str, int] = {}

        # PyTorch named_modules path.
        try:
            named = dict(model.named_modules())
            for n in node_names:
                mod = named.get(n)
                if mod is None:
                    continue
                for attr in ("out_channels", "out_features", "filters", "units", "num_features"):
                    val = getattr(mod, attr, None)
                    if isinstance(val, (int, np.integer)):
                        channels[n] = int(val)
                        break
        except Exception:
            pass

        # Keras layers path.
        try:
            if hasattr(model, "layers"):
                lmap = {getattr(l, "name", ""): l for l in model.layers}
                for n in node_names:
                    if n in channels:
                        continue
                    layer = lmap.get(n)
                    if layer is None:
                        continue
                    for attr in ("filters", "units"):
                        val = getattr(layer, attr, None)
                        if isinstance(val, (int, np.integer)):
                            channels[n] = int(val)
                            break
        except Exception:
            pass

        return channels

    def build_pruning_trace(
        self,
        model: Any,
        score_map: Optional[Dict[str, np.ndarray]] = None,
        masks: Optional[Dict[str, np.ndarray]] = None,
        method_name: str = "unknown",
        candidate_ratio: float = 0.2,
    ) -> Dict[str, Any]:
        """Builds a reusable, presentation-oriented trace artifact for pruning visuals."""
        score_map = score_map or {}
        masks = masks or {}
        ratio = float(np.clip(candidate_ratio, 0.01, 0.95))

        graph = self.adapter.trace_graph(model)
        nodes = graph.get("nodes", {})
        clusters = graph.get("clusters", {})
        order, _, coords = self._depth_layout(nodes)
        ch_map = self._resolve_layer_channels(model, order)

        cluster_ok = {n: True for n in order}
        for _, members in clusters.items():
            valid = [m for m in members if m in masks]
            if len(valid) < 2:
                continue
            ref = np.asarray(masks[valid[0]]).astype(bool)
            for m in valid[1:]:
                same = np.array_equal(ref, np.asarray(masks[m]).astype(bool))
                cluster_ok[m] = same
                cluster_ok[valid[0]] = cluster_ok[valid[0]] and same

        # Build an effective mask set for interpretation: harmonize cluster members.
        effective_masks: Dict[str, np.ndarray] = {}
        for name, m in masks.items():
            effective_masks[name] = np.asarray(m, dtype=bool).reshape(-1)
        for _, members in clusters.items():
            valid = [m for m in members if m in effective_masks]
            if len(valid) < 2:
                continue
            anchor = effective_masks[valid[0]]
            for member in valid[1:]:
                if effective_masks[member].size == anchor.size:
                    effective_masks[member] = anchor.copy()

        layer_stats: Dict[str, Dict[str, Any]] = {}
        for name in order:
            scores = self._to_float_array(score_map.get(name, []))
            mask = effective_masks.get(name, np.array([], dtype=bool))

            orig = int(ch_map.get(name, 0))
            if scores.size:
                orig = max(orig, int(scores.size))
            if mask.size:
                orig = max(orig, int(mask.size))
            if orig <= 0:
                orig = 1

            if scores.size and not mask.size:
                k = max(1, int(round(scores.size * ratio)))
                cand_idx = np.argsort(scores)[:k]
            elif mask.size:
                cand_idx = np.where(~mask)[0]
            else:
                cand_idx = np.array([], dtype=int)

            if mask.size:
                keep = int(np.sum(mask))
            elif scores.size:
                keep = max(1, int(round(scores.size * (1.0 - ratio))))
            else:
                keep = orig

            keep = int(np.clip(keep, 1, orig))
            pruned = int(max(orig - keep, 0))
            keep_ratio = float(keep / max(orig, 1))
            cand_ratio = float(len(cand_idx) / max(orig, 1))

            p10 = float(np.percentile(scores, 10)) if scores.size else None
            p50 = float(np.percentile(scores, 50)) if scores.size else None
            p90 = float(np.percentile(scores, 90)) if scores.size else None

            layer_stats[name] = {
                "type": nodes.get(name, {}).get("type", "unknown"),
                "cluster": nodes.get(name, {}).get("cluster"),
                "inputs": list(nodes.get(name, {}).get("inputs", [])),
                "outputs": list(nodes.get(name, {}).get("outputs", [])),
                "orig_channels": int(orig),
                "kept_channels": int(keep),
                "pruned_channels": int(pruned),
                "keep_ratio": keep_ratio,
                "candidate_ratio": cand_ratio,
                "candidate_indices": [int(i) for i in cand_idx.tolist()[:128]],
                "score_mean": float(np.mean(scores)) if scores.size else None,
                "score_std": float(np.std(scores)) if scores.size else None,
                "score_p10": p10,
                "score_p50": p50,
                "score_p90": p90,
                "cluster_consistent": bool(cluster_ok.get(name, True)),
                "effective_mask_applied": bool(mask.size > 0),
            }

        edges = []
        for src in order:
            for dst in nodes.get(src, {}).get("outputs", []):
                if dst in layer_stats:
                    edges.append({"src": src, "dst": dst})

        return {
            "meta": {
                "method": method_name,
                "candidate_ratio": ratio,
                "node_count": len(order),
                "edge_count": len(edges),
                "cluster_count": len(clusters),
            },
            "order": order,
            "coords": {k: [float(v[0]), float(v[1])] for k, v in coords.items()},
            "clusters": {str(k): list(v) for k, v in clusters.items()},
            "edges": edges,
            "layers": layer_stats,
        }

    def export_pruning_trace(self, trace: Dict[str, Any], path: str = "pruning_trace.json") -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)
        return str(out)

    def summarize_trace_insights(self, trace: Dict[str, Any], max_lines: int = 5) -> List[str]:
        """Generates concise, presenter-friendly insight bullets from a pruning trace."""
        lines: List[str] = []
        layers = trace.get("layers", {})
        if not layers:
            return ["No layer stats found in pruning trace."]

        prunable = [k for k, v in layers.items() if v.get("type") in ("conv2d", "linear")]
        if not prunable:
            return ["No prunable conv/linear layers found in trace."]

        orig_total = int(sum(int(layers[n]["orig_channels"]) for n in prunable))
        kept_total = int(sum(int(layers[n]["kept_channels"]) for n in prunable))
        red_pct = 100.0 * (1.0 - kept_total / max(orig_total, 1))
        lines.append(
            f"Total structural reduction: {orig_total} -> {kept_total} channels ({red_pct:.1f}% removed)."
        )

        ranked_pressure = sorted(
            prunable,
            key=lambda n: float(layers[n]["candidate_ratio"]) + (0.0 if layers[n]["score_mean"] is None else (1.0 - float(layers[n]["score_mean"]))),
            reverse=True,
        )
        top_n = ranked_pressure[0]
        lines.append(
            f"Highest pruning pressure layer: {top_n} "
            f"(cand={layers[top_n]['candidate_ratio']:.1%}, keep={layers[top_n]['keep_ratio']:.1%})."
        )

        clusters = trace.get("clusters", {})
        if clusters:
            cluster_stats = []
            for cid, members in clusters.items():
                valid = [m for m in members if m in layers]
                if not valid:
                    continue
                mean_keep = float(np.mean([layers[m]["keep_ratio"] for m in valid]))
                consistent = all(bool(layers[m].get("cluster_consistent", True)) for m in valid)
                cluster_stats.append((str(cid), len(valid), mean_keep, consistent))
            if cluster_stats:
                worst = sorted(cluster_stats, key=lambda x: x[2])[0]
                lines.append(
                    f"Most compressed dependency cluster: C{worst[0]} "
                    f"(members={worst[1]}, mean keep={worst[2]:.1%}, consistent={worst[3]})."
                )

        head = [n for n in prunable if layers[n].get("type") == "linear" or n == "fc"]
        if head:
            h = head[0]
            lines.append(
                f"Output head status: {h} kept at {layers[h]['keep_ratio']:.1%} "
                f"({layers[h]['orig_channels']}->{layers[h]['kept_channels']})."
            )

        arch_type = "residual" if trace.get("meta", {}).get("cluster_count", 0) > 0 else "sequential/concat"
        lines.append(
            f"Topology summary: {trace.get('meta', {}).get('node_count', len(prunable))} nodes, "
            f"{trace.get('meta', {}).get('edge_count', 0)} edges, "
            f"{trace.get('meta', {}).get('cluster_count', 0)} clusters ({arch_type})."
        )

        return lines[: max(1, int(max_lines))]

    def _edge_traces(
        self,
        trace: Dict[str, Any],
        line_color: str = "#c7c7c7",
        base_width: float = 1.2,
        use_keep_width: bool = False,
    ) -> List[go.Scatter]:
        layers = trace["layers"]
        coords = trace["coords"]
        out: List[go.Scatter] = []
        for e in trace["edges"]:
            src = e["src"]
            dst = e["dst"]
            if src not in coords or dst not in coords:
                continue
            x0, y0 = coords[src]
            x1, y1 = coords[dst]
            width = base_width
            if use_keep_width:
                k0 = float(layers[src]["keep_ratio"])
                k1 = float(layers[dst]["keep_ratio"])
                width = 0.5 + 3.0 * ((k0 + k1) * 0.5)
            out.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color=line_color, width=width),
                    hoverinfo="none",
                    showlegend=False,
                )
            )
        return out

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

    def generate_candidate_discovery_graph(
        self,
        model: Any,
        score_map: Dict[str, np.ndarray],
        masks: Optional[Dict[str, np.ndarray]] = None,
        method_name: str = "apoz",
        candidate_ratio: float = 0.2,
        top_k_layers: int = 14,
    ) -> go.Figure:
        """Graph + table view for candidate channels after metric scoring."""
        trace = self.build_pruning_trace(
            model=model,
            score_map=score_map,
            masks=masks or {},
            method_name=method_name,
            candidate_ratio=candidate_ratio,
        )
        if not trace["order"]:
            return go.Figure()

        order = trace["order"]
        coords = trace["coords"]
        layers = trace["layers"]

        x = [coords[n][0] for n in order]
        y = [coords[n][1] for n in order]
        cand_vals = np.array([layers[n]["candidate_ratio"] for n in order], dtype=np.float64)
        keep_vals = np.array([layers[n]["keep_ratio"] for n in order], dtype=np.float64)
        mu = np.array(
            [layers[n]["score_mean"] if layers[n]["score_mean"] is not None else np.nan for n in order],
            dtype=np.float64,
        )
        if np.all(np.isnan(mu)):
            pressure = cand_vals.copy()
        else:
            valid = ~np.isnan(mu)
            mu_filled = mu.copy()
            mu_filled[~valid] = np.nanmean(mu[valid]) if np.any(valid) else 0.0
            inv_mu = 1.0 - self._normalize01(mu_filled)
            pressure = 0.55 * cand_vals + 0.45 * inv_mu
            # If candidate ratio is uniform (e.g., local pruning), rely more on score pressure.
            if float(np.nanstd(cand_vals)) < 1e-7:
                pressure = inv_mu
        sizes = [12 + 26 * np.clip(np.sqrt(layers[n]["orig_channels"] / 512.0), 0.2, 1.6) for n in order]
        labels = [self._short_name(n) for n in order]

        hover = []
        for n in order:
            d = layers[n]
            hover.append(
                f"<b>{n}</b><br>"
                f"type={d['type']}<br>"
                f"cluster={d['cluster'] if d['cluster'] is not None else '-'}<br>"
                f"orig={d['orig_channels']} ch<br>"
                f"candidate={len(d['candidate_indices'])} ({d['candidate_ratio']:.1%})<br>"
                f"kept={d['kept_channels']} ({d['keep_ratio']:.1%})<br>"
                f"score_p10/p50/p90={d['score_p10'] if d['score_p10'] is not None else '-'} / "
                f"{d['score_p50'] if d['score_p50'] is not None else '-'} / "
                f"{d['score_p90'] if d['score_p90'] is not None else '-'}<br>"
                f"cluster_consistent={d['cluster_consistent']}<br>"
                f"depends_on={d['inputs'][:6]}"
            )

        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.7, 0.3],
            specs=[[{"type": "scatter"}, {"type": "table"}]],
            horizontal_spacing=0.04,
        )

        for tr in self._edge_traces(trace, line_color="rgba(150,160,175,0.40)", base_width=1.3, use_keep_width=False):
            fig.add_trace(tr, row=1, col=1)

        # Overlay cluster links as dashed arcs for dependency emphasis.
        cluster_palette = ["#3b82f6", "#06b6d4", "#8b5cf6", "#f59e0b", "#ef4444", "#10b981"]
        for i, (_, members) in enumerate(trace["clusters"].items()):
            valid = [m for m in members if m in coords]
            if len(valid) < 2:
                continue
            cx = [coords[m][0] for m in valid]
            cy = [coords[m][1] for m in valid]
            fig.add_trace(
                go.Scatter(
                    x=cx,
                    y=cy,
                    mode="lines+markers",
                    line=dict(color=cluster_palette[i % len(cluster_palette)], width=1.0, dash="dot"),
                    marker=dict(size=4, color=cluster_palette[i % len(cluster_palette)]),
                    hoverinfo="none",
                    opacity=0.65,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(
                    size=sizes,
                    color=pressure,
                    colorscale="YlOrRd",
                    cmin=0.0,
                    cmax=max(0.25, float(np.max(pressure))),
                    colorbar=dict(title="Pruning Pressure", x=0.67),
                    line=dict(width=2, color=["#22c55e" if v > 0.75 else "#334155" for v in keep_vals]),
                ),
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        rank_score = {n: float(pressure[i]) for i, n in enumerate(order)}
        ranked = sorted(order, key=lambda n: (rank_score[n], layers[n]["pruned_channels"], -layers[n]["orig_channels"]), reverse=True)[: max(1, int(top_k_layers))]
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Layer", "Orig", "Cand", "Cand %", "Keep %", "Pressure", "Cluster"],
                    fill_color="#0f172a",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[
                        ranked,
                        [layers[n]["orig_channels"] for n in ranked],
                        [len(layers[n]["candidate_indices"]) for n in ranked],
                        [f"{layers[n]['candidate_ratio']:.1%}" for n in ranked],
                        [f"{layers[n]['keep_ratio']:.1%}" for n in ranked],
                        [f"{rank_score[n]:.3f}" for n in ranked],
                        [layers[n]["cluster"] if layers[n]["cluster"] is not None else "-" for n in ranked],
                    ],
                    fill_color="#f8fafc",
                    align="left",
                    font=dict(size=11, color="#0f172a"),
                ),
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
        fig.update_layout(
            template="plotly_white",
            title=(
                f"Candidate Discovery Graph ({method_name.upper()}) | "
                f"nodes={trace['meta']['node_count']} clusters={trace['meta']['cluster_count']}"
            ),
            margin=dict(t=80, l=20, r=20, b=20),
            annotations=[
                dict(
                    x=0.01,
                    y=1.12,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="#334155"),
                    text=(
                        "Node fill = candidate pressure; node border = stronger keep ratio. "
                        "Dashed links mark cluster-constrained dependencies."
                    ),
                )
            ],
        )
        return fig

    def generate_pruning_process_animation(
        self,
        model: Any,
        score_map: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
        method_name: str = "metric",
        candidate_ratio: float = 0.2,
    ) -> go.Figure:
        """Animated narrative of pruning: discovery -> candidates -> cut -> surgery -> final."""
        trace = self.build_pruning_trace(
            model=model,
            score_map=score_map,
            masks=masks,
            method_name=method_name,
            candidate_ratio=candidate_ratio,
        )
        if not trace["order"]:
            return go.Figure()

        order = trace["order"]
        coords = trace["coords"]
        layers = trace["layers"]

        node_x = [coords[n][0] for n in order]
        node_y = [coords[n][1] for n in order]
        names = order
        short_labels = [self._short_name(n) for n in order]
        candidate_vals = np.array([layers[n]["candidate_ratio"] for n in order], dtype=np.float64)
        keep_vals = np.array([layers[n]["keep_ratio"] for n in order], dtype=np.float64)

        base_size = np.array(
            [14 + 20 * np.clip(np.sqrt(layers[n]["orig_channels"] / 512.0), 0.25, 1.6) for n in order],
            dtype=np.float64,
        )
        shrink_size = base_size * (0.55 + 0.45 * keep_vals)

        cluster_palette = ["#6366f1", "#06b6d4", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6"]
        cluster_colors = []
        for n in names:
            cid = layers[n]["cluster"]
            cluster_colors.append(cluster_palette[int(cid) % len(cluster_palette)] if cid is not None else "#94a3b8")

        cut_colors = []
        for n in names:
            d = layers[n]
            if d["pruned_channels"] > 0:
                cut_colors.append(self._blend_hex("#f59e0b", "#ef4444", d["candidate_ratio"] * 1.2))
            else:
                cut_colors.append("#22c55e")

        final_colors = [self._blend_hex("#ef4444", "#22c55e", float(v)) for v in keep_vals]

        hover = []
        for n in names:
            d = layers[n]
            hover.append(
                f"<b>{n}</b><br>"
                f"orig={d['orig_channels']} ch<br>"
                f"candidate={len(d['candidate_indices'])}<br>"
                f"pruned={d['pruned_channels']}<br>"
                f"kept={d['kept_channels']} ({d['keep_ratio']:.1%})<br>"
                f"cluster={d['cluster'] if d['cluster'] is not None else '-'}<br>"
                f"cluster_consistent={d['cluster_consistent']}"
            )

        def node_trace(stage: str) -> go.Scatter:
            if stage == "Discovery":
                colors = cluster_colors
                sizes = base_size
                text = short_labels
            elif stage == "Candidates":
                colors = [self._blend_hex("#cbd5e1", "#f59e0b", v * 1.2) for v in candidate_vals]
                sizes = base_size * (1.0 + 0.15 * candidate_vals)
                text = [f"{lbl}\n{int(round(candidate_vals[i] * 100))}% cand" for i, lbl in enumerate(short_labels)]
            elif stage == "Cut":
                colors = cut_colors
                sizes = base_size * (1.0 - 0.10 * (1.0 - keep_vals))
                text = [f"{short_labels[i]}\n-{layers[names[i]]['pruned_channels']}" for i in range(len(names))]
            elif stage == "Surgery":
                colors = final_colors
                sizes = shrink_size
                text = [f"{short_labels[i]}\n{layers[names[i]]['orig_channels']}->{layers[names[i]]['kept_channels']}" for i in range(len(names))]
            else:
                colors = final_colors
                sizes = shrink_size
                text = [f"{short_labels[i]}\nkeep {int(round(keep_vals[i]*100))}%" for i in range(len(names))]

            return go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=text,
                textposition="top center",
                marker=dict(size=sizes.tolist(), color=colors, line=dict(width=1.5, color="#1e293b")),
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
            )

        stages = ["Discovery", "Candidates", "Cut", "Surgery", "Final"]
        frames = []
        for stage in stages:
            edge_traces = self._edge_traces(
                trace,
                line_color="rgba(120,130,145,0.45)" if stage in ("Discovery", "Candidates") else "rgba(80,90,110,0.45)",
                base_width=1.2,
                use_keep_width=stage in ("Surgery", "Final"),
            )
            frames.append(go.Frame(data=edge_traces + [node_trace(stage)], name=stage))

        fig = go.Figure(
            data=frames[0].data,
            frames=frames,
            layout=go.Layout(
                title=(
                    f"Pruning Process Animation ({method_name.upper()}) | "
                    "Discovery -> Candidates -> Cut -> Surgery -> Final"
                ),
                template="plotly_white",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(
                                label="Play Process",
                                method="animate",
                                args=[
                                    stages,
                                    {"frame": {"duration": 800, "redraw": True}, "transition": {"duration": 250}},
                                ],
                            )
                        ],
                    )
                ],
                annotations=[
                    dict(
                        x=0.01,
                        y=1.10,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=12, color="#334155"),
                        text=(
                            "Candidates highlights metric-selected channels; "
                            "Cut shows removals; Surgery and Final show structural shrink with dependency-preserving edges."
                        ),
                    )
                ],
                margin=dict(t=90, l=20, r=20, b=20),
            ),
        )
        fig.update_layout(
            sliders=[
                {
                    "active": 0,
                    "steps": [{"label": s, "method": "animate", "args": [[s]]} for s in stages],
                }
            ]
        )
        return fig

    def generate_architecture_comparison(
        self,
        model: Any,
        masks: Dict[str, np.ndarray],
        method_name: str = "metric",
    ) -> go.Figure:
        """Side-by-side architecture graph before and after pruning decisions."""
        trace = self.build_pruning_trace(model=model, score_map={}, masks=masks, method_name=method_name, candidate_ratio=0.2)
        if not trace["order"]:
            return go.Figure()

        order = trace["order"]
        coords = trace["coords"]
        layers = trace["layers"]
        ch_orig_total = int(sum(layers[n]["orig_channels"] for n in order if layers[n]["type"] in ("conv2d", "linear")))
        ch_kept_total = int(sum(layers[n]["kept_channels"] for n in order if layers[n]["type"] in ("conv2d", "linear")))
        ch_red_pct = 100.0 * (1.0 - (ch_kept_total / max(ch_orig_total, 1)))

        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.5, 0.5],
            subplot_titles=("Original Architecture", "Pruned Architecture"),
            horizontal_spacing=0.08,
        )

        # Left: Original
        for tr in self._edge_traces(trace, line_color="rgba(130,140,155,0.45)", base_width=1.2, use_keep_width=False):
            fig.add_trace(tr, row=1, col=1)

        x0 = [coords[n][0] for n in order]
        y0 = [coords[n][1] for n in order]
        size0 = [13 + 22 * np.clip(np.sqrt(layers[n]["orig_channels"] / 512.0), 0.2, 1.6) for n in order]
        fig.add_trace(
            go.Scatter(
                x=x0,
                y=y0,
                mode="markers+text",
                text=[self._short_name(n) for n in order],
                textposition="top center",
                marker=dict(size=size0, color="#93c5fd", line=dict(width=1.5, color="#1e3a8a")),
                hovertext=[f"{n}<br>orig={layers[n]['orig_channels']}" for n in order],
                hoverinfo="text",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Right: Pruned
        for tr in self._edge_traces(trace, line_color="rgba(90,110,120,0.45)", base_width=0.9, use_keep_width=True):
            fig.add_trace(tr, row=1, col=2)

        x1 = [coords[n][0] for n in order]
        y1 = [coords[n][1] for n in order]
        keep = np.array([layers[n]["keep_ratio"] for n in order], dtype=np.float64)
        size1 = [10 + 22 * np.clip(np.sqrt(max(layers[n]["kept_channels"], 1) / 512.0), 0.15, 1.5) for n in order]
        color1 = [self._blend_hex("#ef4444", "#22c55e", float(v)) for v in keep]
        fig.add_trace(
            go.Scatter(
                x=x1,
                y=y1,
                mode="markers+text",
                text=[f"{self._short_name(n)}\n{layers[n]['orig_channels']}->{layers[n]['kept_channels']}" for n in order],
                textposition="top center",
                marker=dict(size=size1, color=color1, line=dict(width=1.5, color="#0f172a")),
                hovertext=[
                    f"{n}<br>orig={layers[n]['orig_channels']}<br>kept={layers[n]['kept_channels']}<br>pruned={layers[n]['pruned_channels']}"
                    for n in order
                ],
                hoverinfo="text",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_layout(
            template="plotly_white",
            title=(
                f"Architecture Comparison After Pruning ({method_name.upper()}) | "
                f"channels {ch_orig_total}->{ch_kept_total} ({ch_red_pct:.1f}% reduced)"
            ),
            margin=dict(t=90, l=20, r=20, b=20),
            annotations=[
                dict(
                    x=0.5,
                    y=1.13,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="#334155"),
                    text="Right panel encodes retention in node size and color (green=kept, red=aggressively pruned).",
                )
            ],
        )
        return fig

    def generate_xray_animation(
        self,
        model: Any,
        score_map: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
        method_name: str = "metric",
        candidate_ratio: float = 0.2,
    ) -> go.Figure:
        """Backward-compatible API; now routed to the upgraded process animation."""
        return self.generate_pruning_process_animation(
            model=model,
            score_map=score_map,
            masks=masks,
            method_name=method_name,
            candidate_ratio=candidate_ratio,
        )

        # Legacy implementation retained below for historical reference.
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
        # Third metric fallback:
        # - prefer taylor when available
        # - fallback to apoz for bundled-method-only environments
        from ..pruner.meta_criteria import HybridMetaPruner

        meta = HybridMetaPruner(self.adapter, mode=mode)
        metrics = self.adapter.get_multi_metric_scores(model, loader, ["l1_norm", "mean_abs_act", "taylor", "apoz"])
        if "taylor" in metrics and metrics.get("taylor"):
            third_metric = "taylor"
        elif "apoz" in metrics and metrics.get("apoz"):
            third_metric = "apoz"
        else:
            third_metric = "mean_abs_act"
        prunable_layers = [n for n in node_names if nodes[n].get("type") == "conv2d"]

        contributions: Dict[str, Dict[str, float]] = {}
        for i, layer_name in enumerate(prunable_layers):
            depth = i / (len(prunable_layers) - 1) if len(prunable_layers) > 1 else 0.0
            w_l1, w_act, w_taylor = meta._get_weights(depth)

            m_l1 = np.asarray(metrics.get("l1_norm", {}).get(layer_name, np.ones((1,), dtype=np.float64))).reshape(-1)
            m_act = np.asarray(metrics.get("mean_abs_act", {}).get(layer_name, np.ones((1,), dtype=np.float64))).reshape(-1)
            m_thr = np.asarray(metrics.get(third_metric, {}).get(layer_name, np.ones((1,), dtype=np.float64))).reshape(-1)

            s_l1 = meta._normalize(m_l1)
            s_act = meta._normalize(m_act)
            s_taylor = meta._normalize(m_thr)

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
                "third_metric": third_metric,
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
                    f"{c['third_metric'].upper()} share={c['taylor']:.2%} (w={c['w_taylor']:.2f})"
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
