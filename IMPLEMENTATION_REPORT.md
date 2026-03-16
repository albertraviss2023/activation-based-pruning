# Comprehensive Implementation Report: The Surgical Pruning Framework

## 1. Mathematical Foundations of Importance Scoring

The core of structural pruning is the **Importance Oracle**—a mathematical function that assigns a scalar score $S$ to each filter $F_j$ in a convolutional layer. Filters with the lowest scores are candidates for physical removal.

### 1.1 Weight-Based Norms (Magnitude Pruning)
These methods rely on the hypothesis that the magnitude of a weight is proportional to its influence on the network.
*   **L1-Norm (Manhattan Distance):** Sum of absolute values.
    $$S_j = \frac{1}{|W_j|} \sum_{i=1}^{C_{in}} \sum_{m=1}^{K} \sum_{n=1}^{K} |W_{j,i,m,n}|$$
*   **Literature Reference:** [Li et al., 2017 - Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)

### 1.2 First-Order Taylor Expansion (Gradient-Based)
This heuristic measures the absolute change in the model's loss function $\mathcal{L}$ if a filter's activation $h$ were set to zero.
$$\mathcal{L}(D, h=0) \approx \mathcal{L}(D, h) - \frac{\partial \mathcal{L}}{\partial h}h$$
The importance score is the absolute difference:
$$S_j = \mathbb{E}_{x \in \text{Batch}} \left| \frac{\partial \mathcal{L}}{\partial h_j} \times h_j \right|$$
*   **Literature Reference:** [Molchanov et al., 2019 - Importance Estimation for Neural Network Pruning](https://arxiv.org/abs/1906.10771)

### 1.3 APoZ: Average Percentage of Zeros (Activation-Based)
APoZ measures the sparsity of a filter's activations. A filter that produces zeros across most of the dataset is considered "dead" and redundant.
$$S_j = 1 - \frac{1}{N \cdot H \cdot W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} \mathbb{1}(a_{n,j,h,w} > 0)$$
Where $\mathbb{1}$ is the indicator function and $a$ are post-ReLU activations.
*   **Literature Reference:** [Hu et al., 2016 - Network Trimming: A Data-Driven Neuron Pruning Approach](https://arxiv.org/abs/1607.03250)

---

## 2. Framework Extensibility: Custom Research Extension

The framework allows researchers to define new theories using the `@register_method` API. A prime example implemented in the research suite is **CHIP**.

### 2.1 CHIP: Channel Independence Pruning
CHIP suggests that a channel's importance is defined by the **matrix rank** of its feature maps. High rank implies that the channel captures unique, independent information that cannot be reconstructed from others.

**Mathematical Implementation:**
1.  For each channel $j$, capture activation tensor $\mathcal{A}_j \in \mathbb{R}^{N \times H \times W}$.
2.  Reshape $\mathcal{A}_j$ into a 2D matrix $\mathbf{M}_j \in \mathbb{R}^{N \times (H \cdot W)}$.
3.  Compute the importance score as the matrix rank:
    $$S_j = \text{rank}(\mathbf{M}_j)$$
*   **Literature Reference:** [Sui et al., 2021 - CHIP: Channel Independence-based Pruning](https://arxiv.org/abs/2106.14156)

---

## 3. Global vs. Local Thresholding Algorithms

Once scores are calculated for every layer, the framework executes one of two selection algorithms:

### 3.1 Local Selection (Layer-Wise)
Ensures each layer contributes equally to the pruning goal by applying the ratio independently to each layer's score distribution.

### 3.2 Global Selection (Network-Wide)
Flattens all scores across the entire network into a single vector $\mathbf{V}$, identifying the global threshold $\tau$. This allows the framework to skip pruning critical layers while heavily pruning redundant ones.

---

## 4. Structural Surgery: The Cascading Cut Algorithm

When a filter is removed, the framework must repair the resulting "broken" graph dependencies.

### 4.1 The Dependency Chain
For a sequence $Conv_A \rightarrow BN_A \rightarrow Conv_B$:
1.  **Cut 1 (Output):** Delete row $j$ from $Conv_A$ weights.
2.  **Cut 2 (Normalization):** Delete index $j$ from $BN_A$ mean, variance, and affine parameters.
3.  **Cut 3 (Input):** Delete index $j$ from the **input channel** dimension of $Conv_B$.

### 4.2 Framework-Specific Implementations
*   **PyTorch Backend:** Performs dynamic in-place surgery using tensor slicing and `setattr`. It includes an expansion logic for `Linear` layers following `Conv2d` to map spatial indices correctly.
*   **Keras Backend:** Since Keras graphs are static, the backend uses a functional re-construction approach, rebuilding the model layer-by-layer and injecting the pruned weights.

---

## 5. Diagnostics, Logging & Visualization

### 5.1 Real-Time Research Feedback
To support long research runs in environments like Google Colab, the framework provides:
*   **Explicit Keras Logging:** A custom `ColabLogger` callback that prints epoch-by-epoch status (`loss`, `accuracy`, `val_acc`) to the console to prevent silent timeouts.
*   **Automated Convergence Plots:** Every `train()` call automatically generates a plot of training vs. validation accuracy/loss history.

### 5.2 ROI Diagnostics
The `ParetoAnalyzer` calculates the "Efficiency Frontier" by solving for models that are Pareto-optimal in the (Accuracy, FLOPs) space, calculating the **Speedup Factor** as $\frac{\text{Baseline FLOPs}}{\text{Pruned FLOPs}}$.

---

## 6. Conclusion
The implementation provides a strict separation between **Pruning Theory** (Heuristics) and **Structural Surgery** (Backends). This allows researchers to focus on developing new importance oracles (like CHIP) while the framework handles the complex task of physical architecture rebuilding across dual frameworks.
