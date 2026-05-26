# Hybrid Stack Variability Report

## Purpose

This report explains why the adaptive hybrid pruning stack in ReduCNN should not
be treated as one fixed universal combination of pruning methods. The goal of the
hybrid framework is to discover, per experimental condition and per layer, when a
simple method can replace a more complex method and when complex methods provide
complementary information that should remain in the stack.

The main research position is:

```text
There is no universal best pruning stack. Effective hybrid pruning is
conditional: the best stack depends on architecture, layer band, dataset
complexity, pruning severity, similarity threshold, prune-set agreement, and
measured recovery cost.
```

## What Determines Whether Methods Are Stacked?

In the adaptive hybrid design, methods are stacked only when they provide
meaningfully different pruning information. If a simpler method is sufficiently
similar to a more complex method, the simple method is preferred as a proxy. If
the simple method does not sufficiently represent the complex method, then the
complex method remains in the stack.

The stack therefore changes when the relationship between methods changes.
Method relationships are evaluated using:

- rank correlation between filter/channel importance scores;
- overlap between the filters/channels each method would prune;
- measured simplicity cost, defined as pruning time plus healing time;
- accuracy and efficiency tradeoffs after pruning and recovery.

## Conditions That Can Change the Stack

### 1. Model Architecture

Different CNN architectures produce different layer structures, feature reuse
patterns, and activation behavior.

For example:

- ResNet contains residual blocks and skip connections.
- DenseNet reuses features through dense connectivity.
- VGG is mostly sequential.
- MobileNet uses depthwise separable and inverted residual blocks.

Because of these structural differences, a method that acts as a good proxy in
one model may not act as a good proxy in another. For example, APoZ may proxy a
complex activation method in VGG, while mean absolute activation or L2 norm may
proxy better in ResNet or MobileNet.

### 2. Layer Band

Stacks can vary by layer depth or architectural band.

Early layers often detect simple edges, colors, and textures. These layers may
produce strong agreement between simple magnitude-based or activation-based
methods and more complex methods.

Later layers usually encode more class-specific and semantic information. In
these layers, class-aware, reconstruction-aware, or sensitivity-based methods may
be less replaceable by simple criteria.

A useful reporting pattern is:

```text
In early ResNet blocks, mean_abs_act often proxies complex activation methods.
In later blocks, complex methods remain more distinct and enter the stack.
```

### 3. Dataset

The same model can produce different stacks on different datasets.

Examples:

- CIFAR-10 has fewer classes and relatively simpler class separation.
- CIFAR-100 is more fine-grained and may increase the value of class-aware
  pruning criteria.
- Cat-vs-dog is binary and may change the behavior of class-discriminative
  metrics.

This means that a stack discovered on CIFAR-10 should not automatically be
assumed to be optimal on CIFAR-100 or cat-vs-dog.

### 4. Pruning Ratio

The pruning ratio directly affects method agreement.

At low pruning ratios, methods may agree because they are only removing filters
that are clearly weak. At higher pruning ratios, the methods must decide among
filters that are still moderately useful. This can expose disagreement between
criteria.

Therefore, stacks should be evaluated across pruning ratios such as:

```text
10%, 30%, 50%, and 70%
```

The expected pattern is that simple proxies may be more common at mild pruning
levels, while complex or blended stacks may become more common as pruning becomes
more aggressive.

### 5. Correlation Threshold

The correlation threshold is a direct control variable in the adaptive hybrid
experiment.

At a lower threshold, such as `0.55`, simple methods are allowed to proxy complex
methods more often. At stricter thresholds, such as `0.85` or `0.90`, fewer
simple proxies are accepted and more complex methods may remain in the stack.

This threshold sweep answers:

```text
How stable are method partnerships as the similarity requirement becomes
stricter?
```

The current experiment design begins with:

```python
CORRELATION_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
```

### 6. Prune-Set Overlap Threshold

Rank correlation measures whether two methods order filters similarly overall.
Prune-set overlap asks a more practical question:

```text
Are the methods actually pruning the same filters?
```

This is important because two methods can have high overall rank correlation but
still disagree near the pruning boundary. Since pruning decisions are made at
that boundary, prune-set overlap provides a stricter test of whether a simple
method can safely replace a complex method.

Increasing the prune-set overlap threshold makes the hybrid more conservative.
It will accept fewer simple proxies and keep more complex methods in the stack.

### 7. Simplicity Cost

In this framework, simplicity is not only a theoretical label. It is measured
empirically:

```text
simplicity_time_sec = prune_time_sec + heal_time_sec
```

This means a method is simple if it is cheaper to prune and recover under the
actual experimental setting.

The cost of a method can change depending on:

- model architecture;
- dataset;
- calibration sample size;
- hardware;
- pruning ratio;
- healing or fine-tuning setup.

For this reason, the adaptive hybrid should use the generated efficiency JSON
file from the method-matrix experiments rather than relying on fixed default
costs.

### 8. Healing and Fine-Tuning Behavior

A method may be fast to prune but require longer recovery to regain accuracy.
Another method may be slower to prune but preserve accuracy better and require
less healing.

Therefore, the framework should not rank simplicity using pruning time alone.
The better efficiency definition is:

```text
efficiency = pruning cost + recovery cost + retained accuracy + FLOPs reduction
```

The method-matrix notebooks capture the required fields:

- pruning time;
- healing time;
- baseline accuracy;
- raw pruned accuracy;
- healed accuracy;
- accuracy delta versus baseline;
- FLOPs reduction;
- parameter reduction.

These metrics allow the hybrid method to prefer simple methods only when they
are efficient and effective.

### 9. Hardware and Runtime Environment

The selected stack may change across runtime environments.

Some methods are CPU-heavy, while others benefit more from GPU acceleration.
Activation collection, matrix operations, reconstruction-style methods, and
correlation-based methods can scale differently depending on hardware.

This matters because local CPU smoke tests and Colab GPU experiments may produce
different measured simplicity rankings. Thesis-quality experiments should use
the same target environment, preferably the Colab GPU workflow, for method
matrix generation and adaptive hybrid runs.

### 10. Calibration Sample Size

Activation-based and class-aware methods depend on calibration data.

With too few calibration samples, score rankings may be noisy, and methods may
appear less stable or less correlated. With more calibration samples,
correlation and prune-set overlap may stabilize.

Calibration size should therefore be treated as another experimental condition.
It can help answer:

```text
How much calibration data is needed before adaptive hybrid decisions become
stable?
```

## What Should Be Reported?

The thesis should not report a single fixed statement such as:

```text
The hybrid method is APoZ + NISP + TIS.
```

That would make the method look like a manually stacked ensemble.

Instead, the stronger claim is:

```text
The proposed hybrid framework learns layer-wise method substitution rules. It
identifies when simple methods can safely replace complex methods, and when
complex methods provide complementary information.
```

## Recommended Reporting Table

| Varying condition | What to observe |
|---|---|
| Model architecture | Which simple methods proxy complex methods per architecture |
| Dataset | Whether dataset complexity changes stack composition |
| Layer band | Early, middle, and late layer stack patterns |
| Pruning ratio | When simple proxies stop being sufficient |
| Correlation threshold | Stability of method partnerships |
| Prune-set overlap threshold | Whether ranking similarity matches actual pruning decisions |
| Healing time | Whether fast pruning remains efficient after recovery |
| Calibration size | How much data is needed for stable decisions |
| Hardware/runtime | Whether CPU and GPU environments change measured simplicity |

## Suggested Thesis Framing

A concise thesis framing could be:

```text
Rather than proposing a fixed hybrid pruning formula, this work proposes an
adaptive hybrid selection framework. The framework compares pruning criteria
within each layer and determines whether simpler methods can act as reliable
proxies for more complex methods. If the simple and complex criteria produce
similar rankings and prune similar filters, the simple method is selected. If
not, complementary complex methods are retained in a weighted stack.

The resulting stack is conditional rather than universal. It varies with model
architecture, dataset, layer depth, pruning severity, similarity threshold,
prune-set overlap, calibration data, and measured recovery cost. This conditional
behavior is treated as a strength of the framework because it avoids blindly
stacking methods and instead learns when complexity is justified.
```

## Practical Experiment Implication

The full experimental workflow should be:

1. Run the method-matrix experiments for each model, dataset, pruning ratio, and
   recovery setting.
2. Save the efficiency JSON files.
3. Use those efficiency JSON files as inputs to the adaptive hybrid experiment.
4. Sweep correlation thresholds from `0.55` upward.
5. Analyze which simple methods proxy complex methods per layer band.
6. Report stable method partnerships and conditions where complex methods remain
   necessary.

This workflow makes the hybrid method evidence-driven rather than manually
stacked.
