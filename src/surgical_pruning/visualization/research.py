import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict

def plot_score_distributions(score_maps: Dict[str, Dict[str, np.ndarray]], title_prefix: str = "Model", max_layers: int = 3):
    """
    Plots the distribution of importance scores for different pruning methods.
    Useful for research diagnostics to see if a method produces sparse vs uniform scores.
    """
    if not score_maps:
        return
        
    methods = sorted(score_maps.keys())
    all_layers = sorted({ly for m in methods for ly in score_maps[m].keys()})
    layers_to_plot = all_layers[:max_layers]
    
    if not layers_to_plot:
        return

    fig, axes = plt.subplots(len(layers_to_plot), 1, figsize=(10, 3.5 * len(layers_to_plot)))
    if len(layers_to_plot) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers_to_plot):
        for m in methods:
            if layer in score_maps[m]:
                s = np.asarray(score_maps[m][layer]).reshape(-1)
                ax.hist(s, bins=min(20, max(5, s.size)), alpha=0.35, label=m)
        ax.set_title(f"Score Distribution - {layer}", fontweight='bold')
        ax.set_xlabel("Score (higher => keep)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.suptitle(f"{title_prefix} - Method Score Distributions", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def _spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size != b.size or a.size < 2: return np.nan
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

def plot_rank_correlation(score_maps: Dict[str, Dict[str, np.ndarray]], title_prefix: str = "Model", max_layers: int = 2):
    """
    Plots a heatmap showing how much different pruning methods agree with each other.
    """
    if not score_maps:
        return
        
    methods = sorted(score_maps.keys())
    all_layers = sorted({ly for m in methods for ly in score_maps[m].keys()})
    layers_to_plot = all_layers[:max_layers]
    
    if not layers_to_plot:
        return

    for layer in layers_to_plot:
        mat = np.full((len(methods), len(methods)), np.nan, dtype=float)
        for i, mi in enumerate(methods):
            for j, mj in enumerate(methods):
                if layer in score_maps[mi] and layer in score_maps[mj]:
                    mat[i, j] = _spearman_np(score_maps[mi][layer], score_maps[mj][layer])

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(mat, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                    xticklabels=methods, yticklabels=methods, ax=ax,
                    cbar_kws={'label': 'Spearman Rank Corr'})
        ax.set_title(f"{title_prefix} - Rank Correlation ({layer})", fontweight='bold')
        plt.tight_layout()
        plt.show()


def plot_feature_maps(viz_data: Dict[str, np.ndarray], title_prefix: str = "Model"):
    """
    Visualizes the internal activation feature maps captured during inference.
    Args:
        viz_data: Dictionary containing 'activations' list of numpy arrays.
    """
    acts = viz_data.get("activations", [])
    if not acts:
        print("⚠️ No activations found in viz_data.")
        return

    num_layers = min(len(acts), 3)
    fig, axes = plt.subplots(num_layers, 8, figsize=(15, 2.5 * num_layers))
    
    for i in range(num_layers):
        layer_act = acts[i] # Shape: (C, H, W)
        num_channels = layer_act.shape[0]
        for j in range(8):
            ax = axes[i, j] if num_layers > 1 else axes[j]
            if j < num_channels:
                ax.imshow(layer_act[j], cmap='viridis')
            ax.axis('off')
            if j == 0:
                ax.set_title(f"Layer {i+1}", loc='left', fontweight='bold', fontsize=10)

    plt.suptitle(f"{title_prefix} - First 3 Layers Feature Maps", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Testing Research Visuals with Dummy Data...")
    dummy_scores = {
        "l1_norm": {"conv1": np.random.normal(5, 2, 64)},
        "l2_norm": {"conv1": np.random.normal(5.1, 1.8, 64)},
        "random": {"conv1": np.random.uniform(0, 10, 64)}
    }
    plot_score_distributions(dummy_scores, "Mock Test")
    plot_rank_correlation(dummy_scores, "Mock Test")
