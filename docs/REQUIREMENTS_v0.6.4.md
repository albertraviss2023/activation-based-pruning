# ReduCNN v0.6.4 Iteration Requirements

## 1. Goal
Evolve ReduCNN from a model-specific pruning script into a generalized, architecture-agnostic pruning framework. Introduce advanced graph-based dependency tracking and interactive animations to visualize the structural pruning process in real-time.

## 2. Core Features & Capabilities

### 2.1 Generalized Architecture Compatibility (Graph Topology Classification)
*   **Requirement:** Transition away from hardcoded layer-name matching (e.g., `model.features.0`) to dynamic graph tracing.
*   **Implementation:**
    *   Implement an `ArchitectureClassifier` that analyzes a model's computational graph (using `torch.fx` for PyTorch or Functional API tracing for Keras).
    *   Categorize models into foundational topological classes:
        *   **Class A: Sequential (Direct):** Linear chains of layers (e.g., VGG, AlexNet). Pruning requires straightforward 1:1 input/output channel updates.
        *   **Class B: Residual/Additive (Clustered):** Topologies with split-and-merge operations, specifically element-wise additions (e.g., ResNet, MobileNetV2).
        *   **Class C: Concatenative (Dense):** Topologies merging feature maps via concatenation (e.g., DenseNet).
*   **Impact:** Enables out-of-the-box support for a vast array of modern CNN architectures without requiring manual configuration.

### 2.2 Advanced Dependency Mapping (The "Pruning Cluster" System)
*   **Requirement:** Ensure structural integrity during surgery by identifying and synchronizing interdependent layers.
*   **Implementation:**
    *   During the graph trace, detect operations that impose constraints on tensor shapes (like `Add` nodes).
    *   Group all convolutional layers that feed into a constrained operation into a **Pruning Cluster** and assign them a shared `ClusterID`.
    *   Enforce a **Mask Harmonization** rule: If layer $N$ is in Cluster $X$, it must use the exact same binary pruning mask as all other layers in Cluster $X$, regardless of individual filter importance scores.

### 2.3 Custom Pretrained Model Support (Black-Box Entry Point)
*   **Requirement:** Allow users to import and prune their own pre-trained models seamlessly.
*   **Implementation:**
    *   Introduce a unified API endpoint: `prune_custom_model(model_instance, calibration_dataloader)`.
    *   Implement a `ModelValidator` to ensure the provided model is traceable and compatible with the graph extraction engine.
    *   Skip the baseline training phase automatically when a custom pre-trained model is provided, jumping directly to sensitivity analysis and clustering.
    *   Support saving the pruned custom model back to disk in the native framework format.

### 2.4 Interactive Pruning Animations (The "X-Ray" Visualizer)
*   **Requirement:** Provide an illustrative, animated visualization of the model's graph, highlighting dependencies and the physical pruning process.
*   **Implementation:**
    *   Develop a framework-agnostic `PruningGraph` data structure mapping nodes (layers) and edges (data flow).
    *   Implement an animation engine (e.g., using Plotly for interactive notebooks or Manim for high-quality renders) with four distinct stages:
        1.  **Dependency Discovery:** A visual "radar sweep" that highlights connections and pulses backward from `Add` nodes to visualize the formation of Pruning Clusters (color-coded).
        2.  **Importance Heatmapping:** Nodes transition to a heatmap gradient based on calculated filter importance scores (e.g., Yellow = Vital, Dark Purple = Disposable).
        3.  **The "Physical Shrink" (Surgery):** An animation showing pruned nodes physically deflating (shrinking in size) and connecting edges becoming thinner, symbolizing the removal of parameters and channels.
        4.  **Post-Op Verification:** A final side-by-side comparison showing the "Slim" model overlaid on a ghost image of the original "Fat" model, displaying the new FLOPs and Parameter counts.

## 3. Architecture Updates
*   **`src/reducnn/core/adapter.py`:** Update `FrameworkAdapter` interface to require `trace_graph()` and `classify_architecture()` methods.
*   **`src/reducnn/pruner/surgeon.py`:** Refactor surgery logic to operate on `ClusterID`s rather than sequential layer indices.
*   **`src/reducnn/visualization/animator.py`:** *[NEW MODULE]* Houses the logic for graph extraction, node rendering, and the 4-stage animation pipeline.

## 4. Acceptance Criteria
*   The system can successfully load, trace, cluster, and prune a user-provided custom ResNet-style model without hardcoded rules.
*   The graph visualizer can render the dependency sweep and the shrinking animation in a Jupyter Notebook environment.
*   The pruned custom model maintains structural integrity and can successfully execute a forward pass after surgery.