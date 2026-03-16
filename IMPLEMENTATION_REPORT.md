# Comprehensive Implementation Report: The ReduCNN Framework

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

When a filter is removed, the framework must repair the resulting "broken" graph dependencies. The logic differs significantly between sequential and residual architectures.

### 4.1 Sequential Surgery (VGG-Style)
For a linear sequence $Conv_A \rightarrow BN_A \rightarrow Conv_B$:
1.  **Output Cut:** Delete filter $j$ from $Conv_A$.
2.  **Normalization alignment:** Delete index $j$ from $BN_A$ parameters.
3.  **Input Cut:** Delete index $j$ from the **input channel** dimension of $Conv_B$.
This is handled via a simple topological successor lookup.

### 4.2 Residual Surgery (ResNet-Style)
In branched architectures, a single "cut" has multi-dimensional repercussions.
*   **The Cluster Constraint:** In a residual block $y = f(x) + x$, the number of channels in $f(x)$ must match $x$.
*   **The Solution:** The framework identifies **Residual Clusters**—groups of layers whose outputs are eventually combined.
*   **Harmonization:** If the final convolution of a residual branch is pruned by 30%, the identity shortcut (if it contains a projection convolution) must be pruned by the **exact same 30%** using the **exact same indices**.
*   **FX Tracing:** PyTorch implementation uses `torch.fx` to accurately trace these dependencies through `Add` and `Concat` nodes, ensuring that "input shrinks" are only applied once even if multiple cluster members feed into the same successor.

---

## 5. Dataset Generalization & Auto-Discovery
ReduCNN is designed to be environment-agnostic. 

### 5.1 Dynamic Model Factory
The `get_model(model_type, input_shape, num_classes)` pattern allows the framework to instantiate models for any task:
*   **MNIST:** `(1, 28, 28)` with 10 classes.
*   **CIFAR-100:** `(3, 32, 32)` with 100 classes.
*   **Cats vs Dogs:** `(3, 128, 128)` with 2 classes.

### 5.2 Metadata Inference
The `FrameworkAdapter` derives calibration requirements (batch size, normalization constants) directly from the provided data loaders, allowing for "Plug-and-Prune" workflows where the user only needs to provide their model and a small sample of their data.

### 5.1 Real-Time Research Feedback
To support long research runs in environments like Google Colab, the framework provides:
*   **Explicit Keras Logging:** A custom `ColabLogger` callback that prints epoch-by-epoch status (`loss`, `accuracy`, `val_acc`) to the console to prevent silent timeouts.
*   **Automated Convergence Plots:** Every `train()` call automatically generates a plot of training vs. validation accuracy/loss history.

### 5.2 ROI Diagnostics
The `ParetoAnalyzer` calculates the "Efficiency Frontier" by solving for models that are Pareto-optimal in the (Accuracy, FLOPs) space, calculating the **Speedup Factor** as $\frac{\text{Baseline FLOPs}}{\text{Pruned FLOPs}}$.

---

## 6. Conclusion
The implementation provides a strict separation between **Pruning Theory** (Heuristics) and **Structural Surgery** (Backends). This allows researchers to focus on developing new importance oracles (like CHIP) while the framework handles the complex task of physical architecture rebuilding across dual frameworks.
