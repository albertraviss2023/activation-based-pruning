# ReduCNN v0.6.6 - Hybrid Meta-Pruning Requirements

## 1. Objective
Implement a **Literature-Grounded Hybrid Meta-Pruner** that adaptively blends multiple pruning metrics (Structural, Data-Driven, and Sensitivity-based) based on the relative depth and topological class of the layer. This version aims to maximize accuracy preservation by leveraging the specific strengths of different heuristics at various stages of the network.

## 2. Theoretical Foundation (Literature Review)

The Hybrid Meta-Pruner is built upon four foundational pillars of pruning research:

### 2.1 Early Layer Stability (Weight-Based)
*   **Literature:** *Li et al. (2017), "Pruning Filters for Efficient ConvNets"*
*   **Insight:** Small, dense filters in early layers (capturing edges/textures) are highly sensitive. L1-norm (sum of absolute weights) is a robust proxy for importance here.
*   **v0.6.6 Application:** Prioritize **L1-Norm** for the first **25%** of the network depth.

### 2.2 Middle Layer Redundancy (Activation-Based)
*   **Literature:** *Hu et al. (2016), "Network Trimming: A Data-Driven Neuron Pruning Approach"*
*   **Insight:** Middle layers often exhibit high feature redundancy. **APoZ (Average Percentage of Zeros)** identifies filters that rarely fire across a dataset.
*   **v0.6.6 Application:** Prioritize **APoZ / Mean Absolute Activation** for the **middle 50%** (depth 25% to 75%).

### 2.3 Deep Layer Sensitivity (Gradient-Based)
*   **Literature:** *Molchanov et al. (2019), "Importance Estimation for Neural Network Pruning" (NVIDIA)*
*   **Insight:** As layers approach the loss function, the **First-Order Taylor Expansion** (product of activation and gradient) becomes the most accurate predictor of the delta in final error.
*   **v0.6.6 Application:** Prioritize **Taylor Expansion** for the final **25-30%** of the network.

### 2.4 Structural Dependency Mapping
*   **Literature:** *Fang et al. (2023), "DepGraph: Towards Any-structural Pruning" (CVPR)*
*   **Insight:** Pruning must respect "Dependency Groups" in non-sequential models (ResNet/DenseNet) to maintain tensor shape compatibility across skip connections.
*   **v0.6.6 Application:** Integrate the **v0.6.4 Dependency Graph** to ensure hybrid scores are synchronized across clusters.

---

## 3. Core Features: The "Meta-Pruning" Engine

### 3.1 Adaptive Ensemble Scoring
*   **Requirement:** Calculate a weighted consensus score ($S_{total}$) for every filter.
*   **Formula:** $S_{total} = w_1(d) \cdot S_{L1} + w_2(d) \cdot S_{Act} + w_3(d) \cdot S_{Taylor}$
*   **Implementation:** The weights $w_n(d)$ are dynamic functions of the relative depth ($d \in [0, 1]$).

### 3.2 Transition Modes
*   **Bucket Mode:** Hard thresholds (e.g., Layer 1-4 use L1, Layer 5-10 use Act).
*   **Smooth Blending:** Linear interpolation of weights between zones to avoid "metric shock" at boundary layers.

### 3.3 Conflict Resolution (Voter Logic)
*   **Requirement:** Handle cases where metrics disagree (e.g., L1 says "Keep," Taylor says "Prune").
*   **Implementation:** Implement a "Safety First" policy—if any metric identifies a filter as "Critically Vital" (top 5% score), it is protected from pruning regardless of other scores.

---

## 4. Architecture Updates
*   **`src/reducnn/pruner/meta_criteria.py`:** *[NEW]* Implements the ensemble logic and depth-weighting functions.
*   **`src/reducnn/core/adapter.py`:** Update to allow a single "Calibration Pass" that collects weights, activations, and gradients simultaneously to minimize GPU overhead.
*   **`src/reducnn/visualization/animator.py`:** Enhance the "Heatmap Stage" to show the multi-metric contribution (using a layered or RGB color-blending approach).

## 5. Acceptance Criteria
1.  **Literature Fidelity:** The pruner must demonstrate a shift in metric priority as it traverses from input to output layers.
2.  **Efficiency:** The meta-scoring pass must not exceed 2x the time of a standard Taylor-pruning pass.
3.  **Superiority:** The Hybrid method must outperform any single-metric pruning (L1 or Taylor alone) by at least 0.5% accuracy at high pruning ratios (>50%).
