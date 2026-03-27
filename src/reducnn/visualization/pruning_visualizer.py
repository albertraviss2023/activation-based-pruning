import os
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# Standardized Layer Representation
class LayerVisData(TypedDict):
    layer_name: str
    num_channels: int
    importance_scores: np.ndarray   # shape [C]
    activation_stats: np.ndarray    # shape [C]
    pruned_mask: np.ndarray         # bool mask [C] (True means pruned/removed)

class PruningVisualizer:
    """
    Unified, model-agnostic animation and visualization module for activation-based pruning.
    Produces consistent, high-quality, presentation-ready animations.
    """
    
    def __init__(self, model_name: str, framework: str, experiment_id: str = "default", config: Optional[dict] = None):
        self.model_name = model_name
        self.framework = framework
        self.experiment_id = experiment_id
        self.config = config or {}
        
        # Color Scheme
        self.colors = {
            "baseline": "#1f77b4",  # Blue
            "pruned": "#d62728",    # Red
            "finetuned": "#2ca02c", # Green
            "bg": "#f8f9fa",        # Light background
            "text": "#333333"       # Dark text
        }
        
        # Setup Output Directories scoped by experiment_id
        self.base_out_dir = Path("outputs") / experiment_id
        self.dirs = {
            "activation_flow": self.base_out_dir / "activation_flow",
            "pruning": self.base_out_dir / "pruning",
            "comparisons": self.base_out_dir / "comparisons",
            "recovery": self.base_out_dir / "recovery",
            "structure": self.base_out_dir / "structure"
        }
        
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
        # Matplotlib style tweaks for presentation readiness
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'lines.linewidth': 2,
            'animation.html': 'html5'
        })

    def _get_save_path(self, category: str, filename: str) -> str:
        """Helper to resolve save paths."""
        return str(self.dirs[category] / filename)

    def display_inline(self, filename: str):
        """
        Embeds a saved animation (MP4/GIF) or image (PNG) into a Jupyter Notebook cell.
        """
        from IPython.display import display, Image, Video, HTML
        path = Path(filename)
        if not path.exists():
            print(f"Warning: File not found for inline display: {filename}")
            return

        if path.suffix.lower() == '.mp4':
            # Display video with controls and consistent sizing
            display(Video(filename, embed=True, width=800))
        elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
            display(Image(filename))
        else:
            print(f"Warning: Unsupported format for inline display: {path.suffix}")

    def animate_activation_flow_inline(self, layer_data: LayerVisData, batch_activations: np.ndarray, filename: str = "activation_flow.mp4"):
        """
        High-level wrapper to animate activation flow and immediately display it in the notebook.
        """
        save_path = self._get_save_path("activation_flow", filename)
        self.animate_activation_flow(layer_data, batch_activations, filename=filename)
        self.display_inline(save_path)

    def animate_network_pruning(
        self,
        model: Any,
        loader: Any,
        pruner: Any,
        masks: Dict[str, np.ndarray],
        filename: str = "network_surgery.gif",
        final_hold_frames: int = 25,
    ):
        """
        Highest-level API to generate a full-network 'Network Surgery' animation.
        This follows the PhD-level narrative: Baseline -> Candidates -> Cut -> Collapse -> Recover.
        """
        from .flow_animator import GlobalFlowVisualizer
        from ..backends.factory import get_adapter
        
        # 1. Setup Adapter and data
        config = {'backend': self.framework, 'model_type': self.model_name, 'experiment_id': self.experiment_id}
        adapter = get_adapter(None, config)
        
        print(f"Generating Cinematic Network Surgery Animation for {self.model_name}...")
        
        # 2. Extract necessary components for the pulse
        print("  - Tracing graph topology...")
        graph = adapter.trace_graph(model)
        
        print("  - Capturing global activation pulse...")
        activations = adapter.get_global_activations(model, loader)
        
        print(f"  - Calculating importance scores ({pruner.method})...")
        scores = adapter.get_score_map(model, loader, pruner.method)
        
        # 3. Invoke the Flow Visualizer
        out_path = self._get_save_path("structure", filename)
        
        flow_vis = GlobalFlowVisualizer(
            model_name=self.model_name,
            graph=graph,
            activations=activations,
            scores=scores,
            masks=masks,
            out_path=out_path,
            final_hold_frames=final_hold_frames,
        )
        
        flow_vis.animate()
        self.display_inline(out_path)
        return out_path

    # ==========================================
    # A. Activation Flow Animation
    # ==========================================
    def animate_activation_flow(
        self,
        layer_data: LayerVisData,
        batch_activations: np.ndarray,
        filename: str = "activation_flow.gif",
        prune_ratio: Optional[float] = None,
        threshold_mode: str = "fixed",
    ):
        """
        Animates activation flow as horizontal bars pulsing over time.
        batch_activations shape: [time_steps, num_channels]
        """
        batch_activations = np.asarray(batch_activations, dtype=np.float64)
        if batch_activations.ndim != 2:
            raise ValueError("batch_activations must be a 2D array [time_steps, num_channels].")

        # Use absolute magnitudes so low-activation channels are visible and not clipped by xlim>=0.
        magnitudes = np.abs(batch_activations)

        time_steps, num_channels = magnitudes.shape
        if num_channels != layer_data["num_channels"]:
            raise ValueError("Mismatched channel counts.")

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set up bars
        y_pos = np.arange(num_channels)
        bars = ax.barh(y_pos, magnitudes[0], color=self.colors["baseline"])

        ax.set_title(f"Activation Flow - {self.model_name} ({layer_data['layer_name']})")
        ax.set_xlabel("Activation Magnitude")
        ax.set_ylabel("Channel Index")
        max_val = float(np.max(magnitudes))
        ax.set_xlim(0, max(max_val * 1.1, 1e-6))
        ax.set_ylim(-1, num_channels)

        # Pruning ratio and threshold line for interpretability.
        if prune_ratio is None:
            prune_ratio = float(np.mean(layer_data.get("pruned_mask", np.zeros(num_channels, dtype=bool))))
        prune_ratio = float(np.clip(prune_ratio, 0.0, 0.95))

        # Use batch distribution for fixed threshold so the red/keep split is visible in this animation.
        base_stats = np.abs(np.asarray(layer_data.get("activation_stats", []), dtype=np.float64).reshape(-1))
        if base_stats.size == num_channels and str(threshold_mode).lower().strip() == "stats":
            fixed_threshold = float(np.quantile(base_stats, prune_ratio))
        else:
            fixed_threshold = float(np.quantile(magnitudes.reshape(-1), prune_ratio))

        # Method-selected candidate mask (True means channel selected/pruned by metric policy).
        candidate_mask = np.asarray(layer_data.get("pruned_mask", np.zeros(num_channels, dtype=bool)), dtype=bool).reshape(-1)
        if candidate_mask.size != num_channels:
            candidate_mask = np.zeros(num_channels, dtype=bool)

        threshold_line = ax.axvline(
            x=fixed_threshold,
            color="#6b7280",
            linestyle="--",
            linewidth=1.8,
            label=f"Prune Threshold ({prune_ratio:.0%})",
        )
        ax.legend(loc="lower right")

        def update(frame):
            frame_vals = magnitudes[frame]
            if str(threshold_mode).lower().strip() == "dynamic":
                curr_threshold = float(np.quantile(frame_vals, prune_ratio))
            else:
                curr_threshold = fixed_threshold
            threshold_line.set_xdata([curr_threshold, curr_threshold])

            peak = float(np.max(magnitudes)) + 1e-9
            for i, (bar, val) in enumerate(zip(bars, frame_vals)):
                bar.set_width(val)
                below = bool(val <= curr_threshold)
                is_candidate = bool(candidate_mask[i])
                if below and is_candidate:
                    bar.set_color("#b91c1c")
                elif below:
                    bar.set_color("#ef4444")
                elif is_candidate:
                    bar.set_color("#f59e0b")
                else:
                    bar.set_color(self.colors["baseline"])
                alpha = min(1.0, max(0.30, val / peak))
                bar.set_alpha(alpha)
            return list(bars) + [threshold_line]

        anim = animation.FuncAnimation(fig, update, frames=time_steps, interval=100, blit=True)
        
        save_path = self._get_save_path("activation_flow", filename)
        anim.save(save_path, writer='pillow', fps=10)
        plt.close(fig)
        print(f"Saved Activation Flow Animation: {save_path}")

    # ==========================================
    # B. Channel Importance Visualization
    # ==========================================
    def plot_importance(self, layer_data: LayerVisData, method: str = "apoz", filename: str = "importance.png"):
        """
        Plots sorted channel importance, highlighting the pruning threshold.
        """
        scores = layer_data["importance_scores"]
        mask = layer_data["pruned_mask"]
        
        # Sort indices by score
        sorted_idx = np.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_mask = mask[sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(scores))
        colors = [self.colors["pruned"] if m else self.colors["baseline"] for m in sorted_mask]
        
        bars = ax.bar(x_pos, sorted_scores, color=colors)
        
        # Threshold line (heuristic: highest pruned score)
        if np.any(sorted_mask):
            threshold_val = np.max(sorted_scores[sorted_mask])
            ax.axhline(y=threshold_val, color='gray', linestyle='--', label='Pruning Threshold')
        
        ax.set_title(f"Channel Importance ({method.upper()}) - {self.model_name} / {layer_data['layer_name']}")
        ax.set_xlabel("Sorted Channel Index")
        ax.set_ylabel("Importance Score")
        
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color=self.colors["baseline"], label='Kept Channels'),
            mpatches.Patch(color=self.colors["pruned"], label='Pruned Channels')
        ]
        if np.any(sorted_mask):
            legend_handles.append(plt.Line2D([0], [0], color='gray', linestyle='--', label='Threshold'))
            
        ax.legend(handles=legend_handles)
        
        save_path = self._get_save_path("pruning", filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved Channel Importance Plot: {save_path}")

    # ==========================================
    # C. Pruning Animation (CRITICAL)
    # ==========================================
    def animate_pruning(
        self,
        layer_data: LayerVisData,
        filename: str = "pruning_decision.gif",
        order_mode: str = "decision_then_score",
        final_hold_frames: int = 20,
    ):
        """
        Animates the pruning decision:
        1. Show all channels
        2. Highlight low-importance channels
        3. Fade out / remove them
        """
        scores = layer_data["importance_scores"]
        mask = layer_data["pruned_mask"]
        num_channels = layer_data["num_channels"]
        
        order_mode = str(order_mode).lower().strip()

        if order_mode == "decision_then_score":
            pruned_idx = np.where(mask)[0]
            kept_idx = np.where(~mask)[0]
            pruned_sorted = pruned_idx[np.argsort(scores[pruned_idx])] if pruned_idx.size else np.array([], dtype=int)
            kept_sorted = kept_idx[np.argsort(scores[kept_idx])] if kept_idx.size else np.array([], dtype=int)
            sorted_idx = np.concatenate([pruned_sorted, kept_sorted])
            split_idx = int(len(pruned_sorted))
        else:
            sorted_idx = np.argsort(scores)
            split_idx = int(np.sum(mask[sorted_idx]))

        sorted_scores = scores[sorted_idx]
        sorted_mask = mask[sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(num_channels)
        
        bars = ax.bar(x_pos, sorted_scores, color=self.colors["baseline"])
        
        ax.set_title(f"Pruning Decision Surgery - {layer_data['layer_name']}")
        ax.set_xlabel("Sorted Channel Index")
        ax.set_ylabel("Importance Score")
        ax.set_ylim(0, np.max(scores) * 1.1)

        if split_idx > 0:
            ax.axvline(split_idx - 0.5, color="#6b7280", linestyle="--", linewidth=1.3)
            ax.text(max(split_idx * 0.5 - 1, 0), np.max(scores) * 1.04, "Pruned zone", fontsize=9, color="#b91c1c")
            ax.text(min(split_idx + max((num_channels - split_idx) * 0.5, 1), num_channels - 1), np.max(scores) * 1.04, "Kept zone", fontsize=9, color="#1d4ed8")

        # Identify override-like cases where a pruned channel has score above kept median.
        kept_scores = sorted_scores[~sorted_mask]
        kept_median = float(np.median(kept_scores)) if kept_scores.size else float(np.max(sorted_scores))
        override_mask = np.logical_and(sorted_mask, sorted_scores >= kept_median)
        override_count = int(np.sum(override_mask))
        subtitle = f"Order={order_mode}"
        if layer_data.get("effective_mask_applied"):
            subtitle += " | cluster/effective mask active"
        if override_count > 0:
            subtitle += f" | overrides={override_count}"
        ax.text(0.01, 0.97, subtitle, transform=ax.transAxes, fontsize=9, color="#374151", va="top")
        
        # Animation phases
        # 0-20: Static baseline
        # 21-50: Highlight pruned channels in red
        # 51-90: Fade out / shrink pruned channels
        # 91-110: Static final
        total_frames = 110 + max(0, int(final_hold_frames))
        
        def update(frame):
            if frame < 20:
                pass # baseline
            elif frame < 50:
                # Highlight phase
                for i, b in enumerate(bars):
                    if sorted_mask[i]:
                        b.set_color("#f97316" if override_mask[i] else self.colors["pruned"])
            elif frame < 90:
                # Shrink/fade phase
                progress = (frame - 50) / 40.0
                for i, b in enumerate(bars):
                    if sorted_mask[i]:
                        b.set_alpha(max(0.0, 1.0 - progress))
                        b.set_height(sorted_scores[i] * (1.0 - progress))
                        b.set_color("#f97316" if override_mask[i] else self.colors["pruned"])
            return bars

        anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True)
        
        save_path = self._get_save_path("pruning", filename)
        anim.save(save_path, writer='pillow', fps=20)
        plt.close(fig)
        print(f"Saved Pruning Animation: {save_path}")

    # ==========================================
    # D. Network Compression Animation
    # ==========================================
    def animate_structure_change(self, before_layers: List[LayerVisData], after_layers: List[LayerVisData], filename: str = "network_compression.gif"):
        """
        Animates network macro-structure shrinking.
        """
        if len(before_layers) != len(after_layers):
            raise ValueError("before_layers and after_layers must have the same length.")

        rows = []
        for b, a in zip(before_layers, after_layers):
            rows.append((str(b["layer_name"]), int(b["num_channels"]), int(a["num_channels"])))
        rows.sort(key=lambda x: x[0].lower())

        layer_names = [r[0] for r in rows]
        before_counts = [r[1] for r in rows]
        after_counts = [r[2] for r in rows]

        def _short_name(name: str, max_len: int = 28) -> str:
            if len(name) <= max_len:
                return name
            parts = name.split(".")
            if len(parts) >= 3:
                compact = f"{parts[0]}.{parts[-2]}.{parts[-1]}"
                if len(compact) <= max_len:
                    return compact
            return name[: max_len - 3] + "..."

        display_names = [_short_name(n) for n in layer_names]

        fig_h = min(18, max(6, 2.2 + 0.28 * len(layer_names)))
        fig, ax = plt.subplots(figsize=(13.5, fig_h))
        y_pos = np.arange(len(layer_names))
        
        bars = ax.barh(y_pos, before_counts, color=self.colors["baseline"])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_names, fontsize=8 if len(layer_names) > 24 else 9)
        ax.invert_yaxis()  # Top-down network flow
        ax.set_xlabel("Number of Channels")
        ax.set_title(f"Network Compression - {self.model_name}")
        ax.set_xlim(0, max(before_counts) * 1.1)
        fig.tight_layout()
        
        total_frames = 90
        
        def update(frame):
            if frame < 20:
                pass
            elif frame < 70:
                progress = (frame - 20) / 50.0
                for i, b in enumerate(bars):
                    new_width = before_counts[i] - (before_counts[i] - after_counts[i]) * progress
                    b.set_width(new_width)
            else:
                for i, b in enumerate(bars):
                    b.set_width(after_counts[i])
            return bars

        anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True)
        
        save_path = self._get_save_path("structure", filename)
        anim.save(save_path, writer='pillow', fps=20)
        plt.close(fig)
        print(f"Saved Network Compression Animation: {save_path}")

    # ==========================================
    # E. Fine-Tuning Recovery Animation
    # ==========================================
    def animate_recovery(self, pre_ft_data: LayerVisData, post_ft_data: LayerVisData, filename: str = "recovery_finetuning.gif"):
        """
        Shows activation magnitudes before and after fine-tuning.
        """
        # Ensure we only compare kept channels
        kept_mask = ~pre_ft_data["pruned_mask"]
        pre_stats = pre_ft_data["activation_stats"][kept_mask]
        post_stats = post_ft_data["activation_stats"][kept_mask]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(pre_stats))
        
        bars = ax.bar(x_pos, pre_stats, color=self.colors["pruned"], alpha=0.7, label="Post-Pruning (Pre-FT)")
        
        ax.set_title(f"Fine-Tuning Recovery - {pre_ft_data['layer_name']}")
        ax.set_xlabel("Kept Channel Index")
        ax.set_ylabel("Activation Magnitude")
        ax.set_ylim(0, max(np.max(pre_stats), np.max(post_stats)) * 1.1)
        ax.legend()
        
        total_frames = 80
        
        def update(frame):
            if frame < 20:
                pass
            elif frame < 60:
                progress = (frame - 20) / 40.0
                for i, b in enumerate(bars):
                    current_h = pre_stats[i] + (post_stats[i] - pre_stats[i]) * progress
                    b.set_height(current_h)
                    # Transition color to green
                    b.set_color(self.colors["finetuned"])
            else:
                for i, b in enumerate(bars):
                    b.set_height(post_stats[i])
            return bars

        anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True)
        
        save_path = self._get_save_path("recovery", filename)
        anim.save(save_path, writer='pillow', fps=20)
        plt.close(fig)
        print(f"Saved Recovery Animation: {save_path}")

    # ==========================================
    # F. Multi-Method Comparison
    # ==========================================
    def compare_methods(self, methods_data: List[Dict[str, Any]], layer_name: str, filename: str = "method_comparison.png"):
        """
        Compare which filters each method prunes.
        methods_data: [{"method": "apoz", "data": LayerVisData}, ...]
        """
        num_methods = len(methods_data)
        if num_methods == 0:
            return
            
        num_channels = methods_data[0]["data"]["num_channels"]
        
        fig, ax = plt.subplots(figsize=(12, min(2 + num_methods, 8)))
        
        # Create a heatmap of pruned channels
        # True (pruned) -> 1, False (kept) -> 0
        heatmap_data = np.zeros((num_methods, num_channels))
        method_names = []
        
        for i, md in enumerate(methods_data):
            method_names.append(md["method"].upper())
            heatmap_data[i, :] = md["data"]["pruned_mask"].astype(int)
            
        # Custom colormap: 0=Baseline(Blue), 1=Pruned(Red)
        cmap = LinearSegmentedColormap.from_list("prune_cmap", [self.colors["baseline"], self.colors["pruned"]])
        
        cax = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', interpolation='nearest')
        
        ax.set_yticks(np.arange(num_methods))
        ax.set_yticklabels(method_names)
        ax.set_xlabel("Channel Index")
        ax.set_title(f"Pruning Mask Comparison - {layer_name}")
        
        # Add a custom legend
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color=self.colors["baseline"], label='Kept'),
            mpatches.Patch(color=self.colors["pruned"], label='Pruned')
        ]
        ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        save_path = self._get_save_path("comparisons", filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved Method Comparison Plot: {save_path}")

    # ==========================================
    # G. Sanity Check Helper
    # ==========================================
    def run_sanity_checks(self, before_layers: List[LayerVisData], after_layers: List[LayerVisData]):
        """
        Validates mappings and structural consistency.
        """
        assert len(before_layers) == len(after_layers), "Mismatch in number of layers before/after pruning!"
        
        for bl, al in zip(before_layers, after_layers):
            assert bl["layer_name"] == al["layer_name"], f"Layer name mismatch: {bl['layer_name']} != {al['layer_name']}"
            expected_after = bl["num_channels"] - np.sum(bl["pruned_mask"])
            assert al["num_channels"] == expected_after, \
                f"Channel count mismatch in {bl['layer_name']}. Expected {expected_after}, got {al['num_channels']}"
                
        print("Sanity Checks Passed: Model structure matches pruning masks.")
