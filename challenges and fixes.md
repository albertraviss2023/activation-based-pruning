# Technical Debugging Log: LFPC Pruning Notebook

**Date:** 2026-05-05  
**Focus:** Cats/Dogs pruning notebook, model switching, CHIP scoring runtime, GPU optimization, and experimental validity.

---

## 1. Why This Document Exists

This document records the series of notebook issues that were identified and progressively fixed during the pruning experiments.

The main concerns were:

1. The Cats/Dogs notebook was painfully slow.
2. CIFAR-10 ran much faster than Cats/Dogs.
3. Model switching did not work correctly.
4. Several parts of the notebook were still hardcoded to VGG16 or ResNet18.
5. CHIP scoring consumed roughly 95% of scoring time.
6. GPU optimization was needed.
7. A previous patch accidentally changed data management and introduced Kaggle-authentication problems that did not exist in the original notebook.
8. A deeper methodological concern emerged: if the cost function includes runtime, then optimizing only CHIP could bias the experiments.

---

## 2. Why the CIFAR-10 Notebook Was Much Faster

The CIFAR-10 notebook was not faster simply because it was better written. It was faster because the computational problem was much smaller.

### 2.1 Image Size Difference

CIFAR-10 uses 32 by 32 images.

Cats/Dogs typically uses 224 by 224 images.

That means Cats/Dogs has:

```text
224 × 224 / 32 × 32 = 49× more spatial pixels
```

For convolutional neural networks, especially VGG-like networks, this massively increases the amount of convolutional work.

### 2.2 Data Loading Difference

CIFAR-10 is usually compact and easy to load.

Cats/Dogs often requires:

```text
read JPEG from disk
decode JPEG
resize image
normalize image
batch image
copy image to GPU
run model forward pass
```

If the data is read from Google Drive or another mounted location, data loading and JPEG decoding can become a serious bottleneck.

### 2.3 Cached Results

The CIFAR-10 notebook also appeared to reuse cached singular benchmark results. That means it was not recomputing every expensive pruning benchmark from scratch.

So the runtime comparison was not purely:

```text
CIFAR-10 vs Cats/Dogs
```

It was closer to:

```text
small cached CIFAR-10 experiment vs large high-resolution Cats/Dogs experiment
```

---

## 3. Why the First “Superfast” Patch Was Not Enough

An early speed patch made the notebook much faster by reducing sample sizes, disabling some benchmarks, reducing scoring batches, and turning off healing or fine-tuning.

That was useful for debugging, but not ideal for final scientific reporting.

The user correctly challenged this because reducing the experiment can compromise legitimacy.

The correct principle is:

```text
Do not weaken the experimental protocol just to make the notebook faster.
Instead, optimize the implementation.
```

Legitimate speedups include:

```text
better GPU usage
activation caching
DataLoader reuse
faster vectorized scoring
local disk data loading
mixed precision
TF32
channels-last memory format
```

Less legitimate speedups for final experiments include:

```text
shrinking validation/test samples
turning off benchmark stages
changing pruning ratios
removing methods
disabling healing when healing is part of the protocol
```

---

## 4. Data-Management Mistake That Had To Be Corrected

A later patch accidentally changed the notebook’s data handling by introducing Kaggle download and Kaggle authentication logic.

This caused an error:

```text
You must authenticate before you can call the Kaggle API.
```

This was not requested, and it created a problem that did not exist before.

The original notebook already had a local-drive Cats/Dogs data workflow, so the correct fix was to preserve the original data-management logic and only touch the requested areas:

```text
model switching
VGG16 hardcoding
CHIP scoring speed
GPU optimization
```

Lesson:

```text
When the user asks for targeted notebook fixes, do not modify unrelated parts of the pipeline.
```

---

## 5. Model Switching Problems

The notebook had several architecture-switching problems.

The intended behavior was that changing:

```python
MODEL_TARGET = "resnet18"
```

to:

```python
MODEL_TARGET = "vgg16"
MODEL_TARGET = "densenet121"
MODEL_TARGET = "mobilenet_v2"
```

should automatically update the entire experiment.

### 5.1 Forced Model Reset

One issue was a forced reset pattern similar to:

```python
EXPECTED_MODEL_TARGET = "resnet18"
MODEL_TARGET = EXPECTED_MODEL_TARGET
```

This meant that even if another model was selected, the notebook could silently force the experiment back to ResNet18.

### 5.2 VGG16-Specific Layer Names

Some diagnostic logic still used VGG16-style layers such as:

```python
features.5
features.7
features.10
```

This is unsafe because different architectures expose layers differently.

For example:

```text
VGG16: features.*
ResNet18: layer1.*, layer2.*, layer3.*, layer4.*
DenseNet121: denseblock*, transition*
MobileNetV2: features.* inverted residual blocks
```

### 5.3 Hardcoded Output Names

Some filenames, plot names, or diagnostic labels still included VGG16 even when the active model was not VGG16.

This can create misleading results.

### 5.4 Correct Fix

The selected model should control:

```text
model construction
checkpoint loading
checkpoint filtering
output directories
plot names
CSV names
diagnostic layer selection
prunable layer discovery
dependency-aware pruning behavior
architecture-specific reporting
```

---

## 6. CHIP Runtime Problem

CHIP was reported to take roughly 95% of total scoring time.

That is plausible because CHIP-style channel interaction scoring is expensive, especially for 224 by 224 images.

A slow CHIP path may do something like:

```text
for each layer:
    collect activations
    compute channel interactions
    loop over channels or spatial locations
    repeat across calibration batches
```

On high-resolution images, early convolutional activations are large, so this can become extremely slow.

---

## 7. What Was Done To CHIP

CHIP was not removed.

The slow path was replaced with a GPU-vectorized CHIP-style approximation.

### 7.1 Optimized CHIP Concept

The optimized version captures activations of shape:

```text
[B, C, H, W]
```

Then spatially pools them to:

```text
[B, C]
```

Then computes channel interaction using GPU matrix operations.

Conceptually:

```python
A = activation.mean(dim=(2, 3))      # [B, C]
A = normalize(A, dim=0)
corr = A.T @ A / B                  # [C, C]
score = mean(abs(corr off-diagonal))
```

This uses fast GPU linear algebra instead of expensive Python loops.

### 7.2 Why It Became Very Fast

It became fast because it uses:

```text
spatial pooling
batched GPU matrix multiplication
fewer Python loops
less CPU-GPU transfer
reused scoring loaders
better GPU synchronization
```

### 7.3 Important Scientific Caveat

This is best described as:

```text
fast GPU CHIP-style scoring
```

or:

```text
GPU-accelerated pooled channel-interaction scoring
```

It should not be claimed to be exactly identical to the original CHIP implementation unless equivalence is validated.

---

## 8. GPU and Runtime Optimizations

The requested GPU enhancements were implementation-level improvements, not experimental weakening.

Appropriate optimizations include:

```text
AMP / mixed precision where safe
TF32 on compatible NVIDIA GPUs
channels-last memory format
non-blocking device transfers
cached scoring DataLoader
CUDA synchronization for honest timing
optional torch.compile, disabled by default
```

These improve throughput without changing the dataset, pruning ratios, or core benchmark protocol.

---

## 9. The Key Methodological Question: Can Fast CHIP Bias the Experiment?

Yes.

If the optimization function includes a time element, and CHIP alone is aggressively optimized while other methods remain slow or less optimized, the experiment can become biased.

The optimizer may favor CHIP not because CHIP is inherently the best pruning criterion, but because CHIP was implemented more efficiently.

This is especially important if the cost function looks like:

```text
Cost(method) = α × accuracy_loss + β × FLOPs_cost + γ × scoring_time
```

In that case, implementation engineering becomes part of the objective.

---

## 10. Why This Bias Happens

Suppose two methods have similar pruning quality:

```text
Method A: good accuracy, slow implementation
Method B: good accuracy, highly optimized implementation
```

If scoring time is part of the objective, Method B may win mainly because its implementation is faster.

That may be acceptable if the question is:

```text
Which implemented method is operationally best?
```

But it is not acceptable if the question is:

```text
Which pruning principle is scientifically superior?
```

These are different claims.

---

## 11. Separate the Types of Runtime

The notebook should distinguish at least three different runtime concepts.

### 11.1 Scoring Runtime

Time required to compute pruning scores or masks.

Examples:

```text
APoZ scoring time
L1 scoring time
Taylor scoring time
CHIP scoring time
gradient-based scoring time
```

### 11.2 Pruned Model Inference Runtime

Time required for the final pruned model to perform inference.

This is often the most important runtime for deployment.

### 11.3 End-to-End Pipeline Runtime

Total time including:

```text
data loading
scoring
mask creation
model surgery
fine-tuning or healing
evaluation
reporting
```

These should not be collapsed without explanation.

---

## 12. Recommended Way To Avoid Runtime Bias

### 12.1 Use Two Cost Functions

Use one objective for pruning quality:

```text
QualityCost = α × accuracy_loss + β × FLOPs_ratio + γ × parameter_ratio
```

Use another objective for operational efficiency:

```text
OperationalCost = α × accuracy_loss + β × FLOPs_ratio + γ × scoring_time
```

This keeps scientific pruning quality separate from implementation speed.

### 12.2 Report Scoring Time Separately

Do not hide runtime inside one final score only.

Report:

```text
accuracy after pruning
accuracy drop
FLOPs reduction
parameter reduction
inference latency
scoring time
fine-tuning/healing time
```

### 12.3 Optimize All Methods Fairly

If runtime is included in the objective, then every method should be optimized to a comparable level.

That means:

```text
same calibration batches
same batch size
same device
same precision policy
same DataLoader
same CUDA synchronization
same cached activation policy where applicable
minimal Python loops for all methods
```

If only CHIP is optimized, then the timing result should be labelled as implementation-specific.

### 12.4 Validate Fast CHIP Against Original CHIP

Before using fast CHIP in final reporting, compare it against original CHIP on a smaller subset.

Suggested validation metrics:

```text
Spearman rank correlation of channel scores
Top-k pruning candidate overlap
Jaccard overlap
layer-wise pruning distribution similarity
downstream accuracy after pruning
FLOPs reduction consistency
```

If agreement is high, the fast version is much more defensible.

---

## 13. Recommended Experimental Reporting Strategy

A strong thesis/report structure would include two tracks.

### 13.1 Main Pruning Quality Benchmark

For each model:

```text
VGG16
ResNet18
DenseNet121
MobileNetV2
```

Report:

```text
baseline accuracy
pruned accuracy
accuracy drop
parameter reduction
FLOPs reduction
inference latency
```

This focuses on the quality of the final pruned model.

### 13.2 Scoring Runtime Benchmark

For each method:

```text
APoZ
L1
Taylor
CHIP-original
CHIP-fast
other criteria
```

Report:

```text
scoring time
GPU type
batch size
calibration batches
implementation type
```

This shows operational cost transparently.

### 13.3 Runtime-Aware Stack Search

If the stack optimizer includes runtime, report two variants:

```text
best stack by pruning quality only
best stack by pruning quality + runtime
```

This is more transparent than having one opaque objective.

---

## 14. Recommended Naming for the Fast CHIP Method

Do not simply call it CHIP unless equivalence is proven.

Safer names include:

```text
CHIP-Fast
GPU-CHIP
Pooled-CHIP
CHIP-GPU-Pooled
Fast Channel Interaction Pruning
```

Recommended wording:

```text
We implemented a GPU-accelerated pooled variant of CHIP that estimates channel interaction using spatially aggregated activations and batched covariance/correlation operations. We validate its agreement with the original CHIP implementation using rank correlation, top-k overlap, and downstream pruning performance.
```

---

## 15. Summary of Challenges and Fixes

| Challenge | Why It Mattered | Correct Fix |
|---|---|---|
| Cats/Dogs notebook slow | High-resolution images and JPEG loading are expensive | Optimize GPU/data/scoring implementation |
| CIFAR-10 much faster | CIFAR-10 is smaller and had cached results | Avoid direct runtime comparison |
| First speed patch reduced experiment | Could compromise legitimacy | Use only for debugging, not final reporting |
| Data path changed accidentally | Introduced Kaggle-auth errors | Preserve original local data workflow |
| Model switching broken | Selected architecture did not control all logic | Make `MODEL_TARGET` drive the full notebook |
| VGG16 hardcoding | Diagnostics failed or misled under other models | Select layers dynamically |
| Output names hardcoded | Results could be mislabeled | Use active model name everywhere |
| CHIP took 95% scoring time | Channel interaction scoring was expensive | Add GPU-vectorized pooled CHIP-style scorer |
| Runtime in objective may bias results | Optimized CHIP may win due to engineering | Separate pruning quality from runtime cost |

---

## 16. Immediate Next Steps

1. Run the corrected local-data notebook with:

```python
MODEL_TARGET = "resnet18"
```

2. Confirm no VGG16 labels appear in ResNet18 results.

3. Switch to:

```python
MODEL_TARGET = "vgg16"
MODEL_TARGET = "densenet121"
MODEL_TARGET = "mobilenet_v2"
```

and confirm that each model produces model-specific diagnostics and output files.

4. Add a validation cell comparing original CHIP and fast CHIP on a small subset.

5. Decide whether scoring time should be part of the primary optimization function or only a secondary operational metric.

---

## 17. Suggested Thesis/Report Wording on CHIP

```text
During implementation, CHIP-style channel interaction scoring dominated the scoring phase on high-resolution Cats/Dogs inputs. To make the experiments computationally feasible, we implemented a GPU-accelerated pooled channel-interaction variant that estimates channel redundancy using spatially aggregated activations and batched covariance/correlation operations. Because this optimized implementation may not be numerically identical to the original CHIP formulation, we treat it as a fast CHIP variant and validate its agreement with the original method using rank correlation, top-k overlap, and downstream pruning performance.
```

---

## 18. Suggested Thesis/Report Wording on Runtime Bias

```text
Since the proposed optimization framework may include runtime as part of the cost function, care must be taken to avoid implementation-induced bias. If only one pruning criterion is substantially optimized while others remain naive or CPU-bound, the runtime-aware objective may favor the optimized implementation rather than the underlying pruning principle. Therefore, we report scoring time separately, distinguish pruning quality from operational cost, and where runtime is included in the objective, ensure that all methods are implemented under comparable computational conditions.
```

---

## 19. Bottom Line

The notebook improvements should be framed as two separate contributions:

1. **Engineering contribution:** making the pruning pipeline architecture-switchable, GPU-efficient, and feasible on high-resolution image data.
2. **Methodological contribution:** evaluating whether fast CHIP-style scoring preserves the pruning behavior of original CHIP while reducing computational burden.

This distinction protects the legitimacy of the experiments and makes the thesis argument stronger.
