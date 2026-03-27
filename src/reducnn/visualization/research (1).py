import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List

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

def _topk_mask(scores: np.ndarray, ratio: float) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    keep = max(1, int(round(s.size * (1.0 - ratio))))
    idx = np.argpartition(s, -keep)[-keep:]
    m = np.zeros_like(s, dtype=bool)
    m[idx] = True
    return m

def plot_decision_agreement(score_maps: Dict[str, Dict[str, np.ndarray]],
                            ratio: float = 0.3,
                            title_prefix: str = "Model",
                            max_layers: int = 6):
    """Visualizes agreement of pruning decisions (top-k keep sets) across methods."""
    if not score_maps:
        return
    methods = sorted(score_maps.keys())
    if len(methods) < 2:
        return
    all_layers = sorted({ly for m in methods for ly in score_maps[m].keys()})
    layers = all_layers[:max_layers]
    if not layers:
        return

    jac_sum = np.zeros((len(methods), len(methods)), dtype=np.float64)
    jac_cnt = np.zeros_like(jac_sum)
    layer_agreement: List[float] = []
    layer_labels: List[str] = []

    for layer in layers:
        layer_masks = {}
        for m in methods:
            if layer in score_maps[m]:
                layer_masks[m] = _topk_mask(score_maps[m][layer], ratio=ratio)
        if len(layer_masks) < 2:
            continue

        local_pairs = []
        for i, mi in enumerate(methods):
            for j, mj in enumerate(methods):
                if mi in layer_masks and mj in layer_masks:
                    inter = float(np.logical_and(layer_masks[mi], layer_masks[mj]).sum())
                    union = float(np.logical_or(layer_masks[mi], layer_masks[mj]).sum())
                    jac = inter / max(union, 1.0)
                    jac_sum[i, j] += jac
                    jac_cnt[i, j] += 1.0
                    if i < j:
                        local_pairs.append(jac)
        if local_pairs:
            layer_agreement.append(float(np.mean(local_pairs)))
            layer_labels.append(layer)

    with np.errstate(invalid='ignore', divide='ignore'):
        jac_mean = np.divide(jac_sum, np.maximum(jac_cnt, 1.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(
        jac_mean,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        xticklabels=methods,
        yticklabels=methods,
        ax=ax1,
        cbar_kws={"label": "Mean Jaccard"},
    )
    ax1.set_title("Method Decision Agreement", fontweight="bold")

    if layer_agreement:
        ax2.bar(range(len(layer_agreement)), layer_agreement, color="#4c78a8", alpha=0.9)
        ax2.set_xticks(range(len(layer_agreement)))
        ax2.set_xticklabels(layer_labels, rotation=45, ha="right")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Mean Pairwise Jaccard")
    ax2.set_title("Layer-wise Agreement", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        f"{title_prefix} - Pruning Decision Agreement (ratio={ratio:.2f})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
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

def plot_inference_gallery(images: np.ndarray, true_labels: list, 
                           pred_orig: list, pred_pruned: list, 
                           class_names: list = None, title: str = "Inference Comparison"):
    """
    Plots a grid of images with True, Original, and Pruned labels.
    Args:
        images: Numpy array of shape (N, C, H, W) or (N, H, W, C)
        true_labels: List of ground truth integer labels
        pred_orig: List of predictions from original model
        pred_pruned: List of predictions from pruned model
        class_names: Optional mapping of int -> string label
    """
    n = min(len(images), 8)
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    for i in range(n):
        img = images[i]
        # Transpose if (C, H, W)
        if img.shape[0] == 3 or img.shape[0] == 1:
            img = img.transpose(1, 2, 0)
        
        # De-normalize if needed (heuristic: if mean < 0)
        if np.min(img) < -0.1:
            img = (img * 0.5) + 0.5
        img = np.clip(img, 0, 1)
        
        ax = axes[i]
        ax.imshow(img)
        
        t_idx = true_labels[i]
        o_idx = pred_orig[i]
        p_idx = pred_pruned[i]
        
        t_lab = class_names[t_idx] if class_names else str(t_idx)
        o_lab = class_names[o_idx] if class_names else str(o_idx)
        p_lab = class_names[p_idx] if class_names else str(p_idx)
        
        # Color logic: Red if wrong
        o_color = 'green' if o_idx == t_idx else 'red'
        p_color = 'green' if p_idx == t_idx else 'red'
        
        label_str = f"True: {t_lab}\nOrig: {o_lab}\nPruned: {p_lab}"
        ax.set_title(label_str, fontsize=10, pad=10)
        
        # Visual cues for mismatch
        if o_idx != t_idx or p_idx != t_idx:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
        
        ax.axis('off')
        
    for j in range(n, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
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
