import os
from typing import Dict, Any, List, Optional

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class GlobalFlowVisualizer:
    """
    Schematic visualizer for full-network pruning.

    Key upgrades:
    - Optional delta highlighting against a reference mask set (for method battle).
    - Final-frame hold so GIFs do not feel like they "snap back" instantly.
    - Short, readable layer labels.
    """

    def __init__(
        self,
        model_name: str,
        graph: Dict[str, Any],
        activations: Dict[str, np.ndarray],
        scores: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
        out_path: str = "network_surgery.gif",
        delta_ref_masks: Optional[Dict[str, np.ndarray]] = None,
        delta_only: bool = False,
        total_frames: int = 100,
        final_hold_frames: int = 25,
        show_full_names: bool = False,
    ):
        self.model_name = model_name
        self.nodes = graph.get("nodes", {})
        self.activations = activations
        self.scores = scores
        self.masks = {k: np.asarray(v, dtype=bool).reshape(-1) for k, v in masks.items()}
        self.out_path = out_path
        self.delta_ref_masks = {k: np.asarray(v, dtype=bool).reshape(-1) for k, v in (delta_ref_masks or {}).items()}
        self.delta_only = bool(delta_only)
        self.total_frames = int(max(20, total_frames))
        self.final_hold_frames = int(max(0, final_hold_frames))
        self.phase_frames = int(max(5, self.total_frames // 4))
        self.show_full_names = bool(show_full_names)

        self.colors = {
            "bg": "#ffffff",
            "node_border": "#24292e",
            "active": "#0366d6",
            "weak": "#e1e4e8",
            "candidate": "#f9c513",
            "pruned": "#d73a49",
            "final": "#28a745",
            "text": "#24292e",
            "delta": "#7c3aed",
            "delta_candidate": "#a78bfa",
            "delta_keep": "#4f46e5",
            "muted": "#d1d5db",
        }

        self.layer_names = [
            n
            for n in self.nodes.keys()
            if n in self.activations and n in self.scores and n in self.masks
        ]
        self.layer_names.sort()

        self.nx_graph = nx.DiGraph()
        for n in self.layer_names:
            self.nx_graph.add_node(n)
            outputs = self.nodes[n].get("outputs", [])

            def find_next_prunable(current_outputs, visited=None):
                if visited is None:
                    visited = set()
                found = []
                for o in current_outputs:
                    if o in visited:
                        continue
                    visited.add(o)
                    if o in self.layer_names:
                        found.append(o)
                    elif o in self.nodes:
                        found.extend(find_next_prunable(self.nodes[o].get("outputs", []), visited))
                return found

            for nn in find_next_prunable(outputs):
                self.nx_graph.add_edge(n, nn)

        try:
            from networkx.drawing.nx_agraph import graphviz_layout

            self.pos = graphviz_layout(self.nx_graph, prog="dot")
        except ImportError:
            node_depths = {n: 0 for n in self.layer_names}
            for n in nx.topological_sort(self.nx_graph):
                for pred in self.nx_graph.predecessors(n):
                    node_depths[n] = max(node_depths[n], node_depths[pred] + 1)

            self.pos = {}
            depth_counts = {}
            for n, d in node_depths.items():
                count = depth_counts.get(d, 0)
                self.pos[n] = (d * 6.0, -count * 5.0)
                depth_counts[d] = count + 1

        self._setup_figure()

    def _short_name(self, name: str) -> str:
        if self.show_full_names:
            return name
        if name in ("conv1", "fc"):
            return name
        parts = name.split(".")
        if len(parts) >= 4 and parts[0].startswith("layer") and parts[2] == "downsample" and parts[3] == "0":
            return f"{parts[0]}.{parts[1]}.ds0"
        if len(parts) >= 3 and parts[0].startswith("layer") and parts[2].startswith("conv"):
            return f"{parts[0]}.{parts[1]}.{parts[2]}"
        return name.split(".")[-1][:14]

    def _setup_figure(self, ax=None):
        all_x = [p[0] for p in self.pos.values()]
        all_y = [p[1] for p in self.pos.values()]

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(max(14, len(all_x) * 2), 8))
        else:
            self.ax = ax
            self.fig = ax.get_figure()

        self.ax.set_facecolor(self.colors["bg"])
        self.fig.patch.set_facecolor(self.colors["bg"])
        self.ax.axis("off")

        self.ax.set_xlim(min(all_x) - 3, max(all_x) + 3)
        self.ax.set_ylim(min(all_y) - 4, max(all_y) + 4)

        if ax is None:
            self.ax.set_title(
                f"Architectural Surgery Report: {self.model_name}",
                color=self.colors["text"],
                fontsize=18,
                fontweight="bold",
                pad=30,
            )

        self.draw_objs = {"edges": [], "nodes": {}}

        for u, v in self.nx_graph.edges():
            xu, yu = self.pos[u]
            xv, yv = self.pos[v]
            line = self.ax.annotate(
                "",
                xy=(xv - 1.2, yv),
                xytext=(xu + 1.2, yu),
                arrowprops=dict(arrowstyle="-|>", color="#cbd3da", lw=2, alpha=0.6),
            )
            self.draw_objs["edges"].append({"line": line, "u": u, "v": v})

        for n in self.layer_names:
            x, y = self.pos[n]
            orig_c = len(self.masks[n])
            disp_c = min(orig_c, 24)
            node_w = 2.4
            node_h = 4.0

            header = patches.Rectangle(
                (x - node_w / 2, y + node_h / 2),
                node_w,
                0.6,
                facecolor=self.colors["node_border"],
                zorder=4,
            )
            self.ax.add_patch(header)
            self.ax.text(
                x,
                y + node_h / 2 + 0.2,
                self._short_name(n),
                color="white",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                zorder=5,
            )

            body = patches.Rectangle(
                (x - node_w / 2, y - node_h / 2),
                node_w,
                node_h,
                fill=True,
                facecolor="#f6f8fa",
                edgecolor=self.colors["node_border"],
                lw=1.5,
                zorder=2,
            )
            self.ax.add_patch(body)

            channel_rects = []
            row_h = node_h / disp_c
            for i in range(disp_c):
                r = patches.Rectangle(
                    (x - node_w / 2 + 0.1, y + node_h / 2 - (i + 1) * row_h),
                    node_w - 0.2,
                    row_h * 0.8,
                    fill=True,
                    facecolor=self.colors["weak"],
                    edgecolor="none",
                    zorder=3,
                )
                self.ax.add_patch(r)
                channel_rects.append(r)

            footer_text = self.ax.text(
                x,
                y - node_h / 2 - 0.4,
                f"{orig_c} ch",
                color=self.colors["text"],
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

            self.draw_objs["nodes"][n] = {
                "body": body,
                "header": header,
                "rects": channel_rects,
                "footer": footer_text,
                "pos": (x, y),
                "h": node_h,
                "w": node_w,
                "disp_n": disp_c,
                "orig_n": orig_c,
            }

    def _interpolate_color(self, c1, c2, t):
        t = max(0.0, min(1.0, float(t)))
        c1 = c1.lstrip("#")
        c2 = c2.lstrip("#")
        r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
        r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
        return f"#{int(r1 + (r2 - r1) * t):02x}{int(g1 + (g2 - g1) * t):02x}{int(b1 + (b2 - b1) * t):02x}"

    def update(self, frame):
        if frame >= self.total_frames:
            phase = 4
            progress = 1.0
        else:
            phase = min(4, (frame // self.phase_frames) + 1)
            progress = (frame % self.phase_frames) / float(self.phase_frames)

        artists = []
        for name in self.layer_names:
            node = self.draw_objs["nodes"][name]
            mask_orig = self.masks[name]
            idx_subset = np.linspace(0, len(mask_orig) - 1, node["disp_n"], dtype=int)
            mask = mask_orig[idx_subset]

            scores_raw = np.asarray(self.scores[name]).reshape(-1)
            scores = scores_raw[idx_subset]
            denom = (np.max(scores) - np.min(scores)) + 1e-8
            scores_n = (scores - np.min(scores)) / denom

            ref = self.delta_ref_masks.get(name)
            if ref is not None and len(ref) == len(mask_orig):
                delta_mask = (mask_orig != ref)[idx_subset]
            else:
                delta_mask = np.zeros_like(mask, dtype=bool)
            delta_count = int(np.sum(mask_orig != ref)) if ref is not None and len(ref) == len(mask_orig) else 0

            for i, rect in enumerate(node["rects"]):
                rect.set_width(node["w"] - 0.2)

                c_logic = self._interpolate_color(self.colors["weak"], self.colors["active"], scores_n[i])
                if delta_mask[i]:
                    c_logic = self._interpolate_color(self.colors["weak"], self.colors["delta"], max(0.35, scores_n[i]))

                if self.delta_only and not delta_mask[i] and phase >= 2:
                    rect.set_facecolor(self.colors["muted"])
                    artists.append(rect)
                    continue

                if phase == 1:
                    rect.set_facecolor(c_logic)
                elif phase == 2:
                    if mask[i] == 0:
                        pulse = (np.sin(frame * 0.5) + 1) / 2
                        target = self.colors["delta_candidate"] if delta_mask[i] else self.colors["candidate"]
                        rect.set_facecolor(self._interpolate_color(c_logic, target, pulse))
                    else:
                        rect.set_facecolor(c_logic)
                elif phase == 3:
                    if mask[i] == 0:
                        start = self.colors["delta_candidate"] if delta_mask[i] else self.colors["candidate"]
                        rect.set_facecolor(self._interpolate_color(start, self.colors["pruned"], progress))
                        rect.set_width((node["w"] - 0.2) * (1 - progress))
                    else:
                        rect.set_facecolor(self.colors["delta_keep"] if delta_mask[i] else self.colors["active"])
                else:
                    if mask[i] == 1:
                        rect.set_facecolor(self.colors["delta_keep"] if delta_mask[i] else self.colors["final"])
                    else:
                        rect.set_facecolor(self.colors["bg"])
                        rect.set_width(0)
                artists.append(rect)

            if phase == 4:
                ratio = float(np.mean(mask_orig))
                target_h = node["h"] * max(0.3, ratio)
                curr_h = node["h"] - (node["h"] - target_h) * progress
                node["body"].set_height(curr_h)
                node["body"].set_y(node["pos"][1] - curr_h / 2)
                node["header"].set_y(node["pos"][1] + curr_h / 2)
                if delta_count > 0:
                    node["footer"].set_text(f"{int(np.sum(mask_orig))} ch | Δ{delta_count}")
                else:
                    node["footer"].set_text(f"{int(np.sum(mask_orig))} ch")
                artists.extend([node["body"], node["header"], node["footer"]])

        return artists

    def animate(self):
        n_frames = self.total_frames + self.final_hold_frames
        anim = animation.FuncAnimation(self.fig, self.update, frames=n_frames, interval=50, blit=True)
        print("Rendering Architectural Schematic Surgery...")
        anim.save(self.out_path, writer="pillow", fps=20)
        plt.close(self.fig)
        print(f"Animation saved to: {self.out_path}")


class GlobalMethodComparator:
    """
    Compares two pruning strategies.

    Delta mode highlights where method A and B disagree at channel level.
    """

    def __init__(
        self,
        model_name: str,
        graph: Dict[str, Any],
        activations: Dict[str, np.ndarray],
        method_a_data: Dict[str, Any],
        method_b_data: Dict[str, Any],
        out_path: str = "method_battle.gif",
        delta_mode: bool = True,
        delta_only: bool = False,
        total_frames: int = 100,
        final_hold_frames: int = 25,
    ):
        self.model_name = model_name
        self.out_path = out_path
        self.method_a_name = method_a_data["name"]
        self.method_b_name = method_b_data["name"]
        self.delta_mode = bool(delta_mode)

        ref_a = method_b_data["masks"] if self.delta_mode else None
        ref_b = method_a_data["masks"] if self.delta_mode else None

        self.viz_a = GlobalFlowVisualizer(
            model_name,
            graph,
            activations,
            method_a_data["scores"],
            method_a_data["masks"],
            delta_ref_masks=ref_a,
            delta_only=delta_only,
            total_frames=total_frames,
            final_hold_frames=final_hold_frames,
        )
        self.viz_b = GlobalFlowVisualizer(
            model_name,
            graph,
            activations,
            method_b_data["scores"],
            method_b_data["masks"],
            delta_ref_masks=ref_b,
            delta_only=delta_only,
            total_frames=total_frames,
            final_hold_frames=final_hold_frames,
        )
        plt.close(self.viz_a.fig)
        plt.close(self.viz_b.fig)
        self._setup_comparison_figure()

    def _mean_disagreement(self, masks_a: Dict[str, np.ndarray], masks_b: Dict[str, np.ndarray]) -> float:
        common = sorted(set(masks_a.keys()).intersection(masks_b.keys()))
        vals = []
        for n in common:
            ma = np.asarray(masks_a[n], dtype=bool).reshape(-1)
            mb = np.asarray(masks_b[n], dtype=bool).reshape(-1)
            if len(ma) != len(mb):
                continue
            vals.append(float(np.mean(ma != mb)))
        return float(np.mean(vals)) if vals else 0.0

    def _setup_comparison_figure(self):
        self.fig, (self.ax_a, self.ax_b) = plt.subplots(2, 1, figsize=(16, 12))
        self.fig.patch.set_facecolor("#ffffff")

        disagreement = self._mean_disagreement(self.viz_a.masks, self.viz_b.masks)
        subtitle = f"Channel disagreement: {disagreement:.1%}"
        if self.delta_mode:
            subtitle += " (delta-highlight enabled)"

        for ax, name, viz in [
            (self.ax_a, self.method_a_name, self.viz_a),
            (self.ax_b, self.method_b_name, self.viz_b),
        ]:
            ax.set_title(f"PRUNING STRATEGY: {name} | {subtitle}", color="#24292e", fontsize=13, fontweight="bold")
            viz._setup_figure(ax=ax)

    def update(self, frame):
        return self.viz_a.update(frame) + self.viz_b.update(frame)

    def animate(self):
        n_frames = self.viz_a.total_frames + self.viz_a.final_hold_frames
        anim = animation.FuncAnimation(self.fig, self.update, frames=n_frames, interval=50, blit=True)
        print("Rendering Comparative Strategy Battle...")
        anim.save(self.out_path, writer="pillow", fps=20)
        plt.close(self.fig)
        print(f"Comparison saved to: {self.out_path}")
