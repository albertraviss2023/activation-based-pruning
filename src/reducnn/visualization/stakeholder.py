import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
from .persistence import persist_matplotlib_figure

def plot_layer_sensitivity(masks: Dict[str, np.ndarray], title_prefix: str = "Model"):
    """
    Visualizes the percentage of filters kept in each layer.
    """
    if not masks:
        print("⚠️ No masks provided.")
        return
        
    layers_list = sorted(masks.keys())
    keep_ratios = [float(np.mean(masks[l]) * 100) for l in layers_list]
    n_layers = len(layers_list)

    fig_w = float(np.clip(0.22 * n_layers + 8.0, 12.0, 32.0))
    fig_h = 6.5 if n_layers > 80 else 5.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = np.arange(n_layers)
    colors = plt.cm.RdYlGn(np.array(keep_ratios) / 100.0)
    bars = ax.bar(x, keep_ratios, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    avg_keep = float(np.mean(keep_ratios))
    ax.axhline(y=avg_keep, color='blue', linestyle='--', label=f'Avg Keep: {avg_keep:.1f}%')
    ax.set_title(f"{title_prefix} - Layer-Wise Pruning Sensitivity", fontsize=15, fontweight='bold')
    ax.set_ylabel("Filters Kept (%)", fontsize=12)

    # Dense models can have 100+ conv layers; render only a subset of labels.
    if n_layers <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels(layers_list, rotation=45, ha='right', fontsize=9)
    else:
        step = max(1, int(np.ceil(n_layers / 28)))
        ticks = x[::step]
        ax.set_xticks(ticks)
        ax.set_xticklabels([layers_list[i] for i in ticks], rotation=60, ha='right', fontsize=8)
        ax.set_xlabel(f"Layer index (showing every {step}th label out of {n_layers})", fontsize=10)

    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    # Only annotate bar values when the plot is small enough to stay readable.
    if n_layers <= 35:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 1.0,
                f"{h:.1f}%",
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold',
            )
    
    plt.tight_layout()
    persist_matplotlib_figure(fig, f"{title_prefix}_layer_sensitivity")
    plt.show()

def plot_metrics_comparison(b_stats, p_stats):
    """
    Plots a side-by-side comparison of Parameters, FLOPs, and Accuracy.
    Args:
        b_stats: Tuple (FLOPs, Params) or Dict {'FLOPs': f, 'Params': p, 'Acc': a}
        p_stats: Tuple (FLOPs, Params) or Dict {'FLOPs': f, 'Params': p, 'Acc': a}
    """
    # Standardize input to dictionaries
    def standardize(s):
        if isinstance(s, dict):
            return s
        # If tuple, assume (FLOPs, Params)
        return {'FLOPs': s[0], 'Params': s[1], 'Acc': 0.0}

    b_dict = standardize(b_stats)
    p_dict = standardize(p_stats)

    labels = ['Params (M)', 'FLOPs (M)', 'Accuracy (%)']
    b_vals = [b_dict['Params']/1e6, b_dict['FLOPs']/1e6, b_dict.get('Acc', 0.0)]
    p_vals = [p_dict['Params']/1e6, p_dict['FLOPs']/1e6, p_dict.get('Acc', 0.0)]
    
    x = np.arange(len(labels)); w = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    r1 = ax.bar(x - w/2, b_vals, w, label='Baseline', color='gray', alpha=0.7)
    r2 = ax.bar(x + w/2, p_vals, w, label='Pruned', color='salmon', alpha=0.9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Value')
    ax.legend()
    ax.set_title("Final Metrics Comparison", fontweight='bold', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add labels on top of bars
    for r in [r1, r2]:
        for bar in r:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.2f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    persist_matplotlib_figure(fig, "final_metrics_comparison")
    plt.show()
    
    reduction_pct = (1 - p_dict['FLOPs']/b_dict['FLOPs']) * 100
    acc_drop = b_dict.get('Acc', 0) - p_dict.get('Acc', 0)
    print(f"🚀 Business Impact: Model is {b_dict['FLOPs']/p_dict['FLOPs']:.2f}x faster (FLOPs reduced by {reduction_pct:.1f}%).")
    if acc_drop != 0:
        print(f"📉 Accuracy Delta: {acc_drop:+.2f}%")

def plot_training_history(history: Dict[str, list], title: str = "Training History"):
    """
    Plots training and validation accuracy/loss over epochs.
    """
    if not history or not history.get('train_loss'):
        print(f"ℹ️ Training history for '{title}' is empty. Skipping plot.")
        return

    epochs = np.arange(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    if history.get('train_acc'):
        ax1.plot(epochs, history['train_acc'], label='Train Acc', marker='.')
    if history.get('val_acc'):
        ax1.plot(epochs, history['val_acc'], label='Val Acc', marker='.')
    ax1.set_title(f"{title} - Accuracy")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history['train_loss'], label='Train Loss', marker='.')
    if history.get('val_loss'):
        ax2.plot(epochs, history['val_loss'], label='Val Loss', marker='.')
    ax2.set_title(f"{title} - Loss")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    persist_matplotlib_figure(fig, f"{title}_training_history")
    plt.show()


if __name__ == "__main__":
    # Isolated Troubleshooting / Testing Block
    print("Testing Stakeholder Visuals with Dummy Data...")
    
    dummy_masks = {
        "conv1": np.array([True, True, False, True]), # 75%
        "conv2": np.array([True, False, False, False]), # 25%
        "conv3": np.array([True, True, True, True]) # 100%
    }
    plot_layer_sensitivity(dummy_masks, "Mock")
    
    # (FLOPs, Params)
    plot_metrics_comparison((100e6, 50e6), (60e6, 25e6))
