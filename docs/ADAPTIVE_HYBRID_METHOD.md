# Adaptive Hybrid Pruning Strategy

This document describes the revised hybrid pruning idea for ReduCNN.

The goal is not to stack pruning methods blindly. The goal is to decide, per
layer, whether a cheaper pruning criterion can replace a more expensive one
because both criteria produce similar channel rankings.

## Core Idea

For each prunable layer, ReduCNN can compute several method score vectors:

```text
l1_norm, mean_abs_act, apoz, taylor, chip, custom_tis, custom_nisp, ...
```

Each vector ranks channels by importance. This directly follows the pruning
criteria framing in Huang et al., who analyze whether criteria produce almost
identical ranks of filters' importance scores and therefore similar pruned
structures.

In ReduCNN, two methods are considered similar in a layer only when both are
true:

1. Their full channel rankings have high Spearman rank correlation.
2. Their bottom-k prune sets overlap strongly at the current pruning ratio.

This second check matters because pruning only removes the least important
channels. Two criteria can have high global rank correlation but still disagree
near the pruning threshold.

The adaptive hybrid now allows simple methods to form their own stack when they
are the compatible candidates for a scope. This is important evidence rather
than a corner case: if simple methods agree with one another, the experiment
should report whether the simple stack is competitive against each individual
simple method.

The dataset-agnostic hybrid notebook keeps pruning policies separate by
compatibility, not by assigning each method to exactly one scope. A method can be
compatible with both `local` and `global` thresholding. The experiment therefore
builds one stack from all local-compatible candidates and one stack from all
global-compatible candidates. This prevents accidental exclusion of simple
methods from global experiments while still avoiding an unlabelled mixture of
different pruning policies.

If a simple method and a complex method agree on the filters to prune, use the
simpler method. The simplicity ranking is already defined before the adaptive
choice is made, so the layer decision does not need to rediscover simplicity by
adding more timing terms. Layer-level reports can record the prune-set
agreement, selected method, channel/FLOPs impact, and score time for that layer.
Accuracy retention is model-level evidence only, measured after the full
pruning/healing run, so it must not be interpreted as a layer-wise accuracy
change.

If the methods disagree, keep more than one signal through a weighted blend.

## Working Hypothesis

The hybrid method tests the following thesis hypothesis:

```text
If simpler pruning criteria select nearly the same prune candidates as more
complex criteria in some layer bands, then ReduCNN can replace the complex
criteria with simpler proxies in those layers and build a layer-wise hybrid
stack that is more efficient than applying one complex method end to end.
```

Efficiency is evaluated empirically, not assumed. A hybrid run should be
compared against the individual methods that appear in its selected stacks using:

- pruning/scoring time;
- healing/fine-tuning time;
- total simplicity time, `prune_time_sec + heal_time_sec`;
- healed accuracy and accuracy change against baseline;
- FLOPs reduction;
- parameter reduction.

Layer selection is therefore not a "fastest method wins" rule. Agreement comes
first, then simplicity. The utility-style evidence below is kept for reporting
and tie-breaks, not as a replacement for the simpler-over-complex rule:

```text
utility =
  accuracy-retention factor
  * FLOPs-reduction factor
  * parameter-reduction factor
  / measured simplicity cost
```

The FLOPs term is deliberate. A pruning method that is quick to score but gives
weak compression is not automatically a good proxy. Conversely, a method that is
slightly slower can still be selected when it gives better accuracy retention
and stronger FLOPs reduction.

The comparison questions are:

```text
Does the adaptive hybrid keep the accuracy/FLOPs benefits of useful methods
while reducing pruning and recovery cost by replacing redundant complex methods
with simpler correlated methods layer by layer?

Which local-compatible methods work best together, which global-compatible
methods work best together, and do those stacks outperform the individual
methods after scoring, pruning, and healing time are all counted?
```

The result may be conditional. If a complex method is not correlated with any
simpler method in a layer band, the hybrid should keep that complex method or
blend it with complementary signals. If a simple proxy is fast but harms healed
accuracy or gives weak FLOPs reduction, the comparison table should expose that
trade-off.

## Why This Is a Better Hybrid

A static hybrid says:

```text
score = w1 * method_a + w2 * method_b + w3 * method_c
```

That is easy to implement, but it does not explain why the selected methods
belong together.

The adaptive hybrid asks:

```text
In this layer, are these methods actually making similar decisions?
If yes, use the simplest one.
If no, combine the complementary signals.
```

This turns the hybrid method into a layer-wise decision policy.

## Layer-Wise Similarity

Raw pruning scores may use different scales, so ReduCNN first converts each
method score vector into rank percentiles:

```text
rank_i = rank(score_i) / (num_channels - 1)
```

Then it computes Spearman-style rank correlation between method pairs:

```text
corr(method_a, method_b, layer_l)
```

If:

```text
abs(corr(simple_method, complex_method)) >= threshold
```

then the simple method is treated as a proxy for the complex method in that
layer.

For exploratory experiments, the threshold is swept from permissive to stricter
values:

```text
0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
```

The production default can remain stricter, for example `0.90`, after the
sensitivity study shows which threshold behaves best.

ReduCNN also compares the bottom-k channel sets:

```text
prune_set(method, layer) = k channels with lowest importance scores
overlap = |prune_set(a) intersect prune_set(b)| / k
```

The default top-k overlap threshold is `0.80`.

The pruning ratio is part of the similarity test. A pair of methods may agree
on the bottom 30% of filters and disagree on the bottom 50%. For this reason,
the adaptive notebook now records whether the efficiency evidence was measured
at the same pruning ratio as the hybrid run. If the method matrix was generated
only at `0.30`, then `0.50` and `0.60` hybrid runs are marked as using nearest
or cross-ratio evidence. Those runs are useful for exploration, but the most
defensible thesis comparison is the ratio for which the individual method
matrix was actually measured.

## Cost-Aware Selection

## What "Simple" Means

In this thesis, "simple" does not mean mathematically unsophisticated. It means
cheaper to compute during a pruning run.

ReduCNN defines simplicity using a cost score based on:

- whether the method needs only weights;
- whether it needs forward activation collection;
- whether it needs backward gradients;
- whether it needs class-wise passes or ablations;
- whether it needs cross-layer reconstruction, propagation, or similarity
  matrices;
- observed pruning-time measurements when available.

Lower cost means simpler.

The strongest definition of cost is measured cost from the experiment notebooks:

```text
simplicity_time_sec = prune_time_sec + heal_time_sec
```

This is the preferred cost signal because it includes both the time needed to
score/prune and the time needed to recover accuracy after pruning.

For thesis-grade adaptive hybrid runs, this measured ranking file should be
created before the hybrid notebook is run. The fallback costs are useful only for
checking that the notebook executes; they should not be used as evidence for the
research claim.

Each method is assigned a default approximate cost:

```text
l1_norm          1.0   weights only
l2_norm          1.2   weights only
mean_abs_act     2.0   forward activations
apoz             2.0   forward activations
custom_entropy   2.5   activation distribution
custom_hrank     3.5   activation matrix rank
chip             4.0   activation nuclear-norm change
custom_reprune   4.5   kernel similarity / clustering
taylor           5.0   forward + backward gradients
custom_tis       5.0   class-wise Taylor thresholding
custom_nisp      5.5   cross-layer importance propagation
custom_thinet    6.0   next-layer reconstruction damage
custom_senpis    6.5   class-wise ablation / attenuation
```

The adaptive hybrid searches for efficient methods that cover expensive methods.
Coverage means high rank correlation and high prune-set overlap with a
higher-cost method. The replacement decision is then ranked by measured utility:
time, healed accuracy, FLOPs reduction, and parameter reduction.

Example:

```text
Layer conv3:
  corr(l1_norm, taylor) = 0.94
  prune-set overlap(l1_norm, taylor) = 0.88
  cost(l1_norm) = 1.0
  cost(taylor) = 5.0

Decision:
  Use l1_norm for conv3 because it gives nearly the same channel ordering as
  Taylor at much lower scoring cost.
```

## When Methods Disagree

If no cheap method can proxy for an expensive method, ReduCNN falls back to a
weighted blend:

```text
score = sum(weight_m * rank(method_m))
```

The weight is larger when:

- the method has high score dispersion, meaning it can distinguish channels;
- the method has stronger measured efficiency utility, including lower
  simplicity time, better healed accuracy, and better FLOPs reduction.

This means the method still benefits from complementary criteria when the
rankings disagree.

## ReduCNN Configuration

Use adaptive mode:

```python
config = {
    "method": "hybrid",
    "meta_mode": "adaptive",
    "hybrid_metric_pool": ["l1_norm", "mean_abs_act", "apoz", "taylor", "chip"],
    "hybrid_correlation_threshold": 0.90,
    "hybrid_topk_overlap_threshold": 0.80,
}
```

The method pool can include custom registered methods:

```python
config = {
    "method": "hybrid",
    "meta_mode": "adaptive",
    "hybrid_metric_pool": [
        "l1_norm",
        "mean_abs_act",
        "apoz",
        "custom_tis",
        "custom_nisp",
        "custom_senpis",
    ],
}
```

Custom costs can be supplied:

```python
config = {
    "hybrid_method_costs": {
        "l1_norm": 1.0,
        "mean_abs_act": 2.0,
        "apoz": 2.0,
        "custom_tis": 5.0,
        "custom_senpis": 7.0,
    }
}
```

Or load measured costs from an efficiency JSON produced by the notebooks:

```python
config = {
    "method": "hybrid",
    "meta_mode": "adaptive",
    "backend": "pytorch",
    "model_type": "resnet18",
    "dataset": "cifar-10",
    "hybrid_efficiency_json_path": "outputs/custom_method_matrix/notebook_run_resnet18_balanced_YYYYMMDD_HHMMSS_efficiency.json",
}
```

When this file is provided, ReduCNN uses the median measured
`simplicity_time_sec` per method as the simplicity cost. By default the costs
are normalized so the fastest method has cost `1.0`.

In `experiments_adaptive_hybrid_dataset_agnostic.ipynb`, keep
`REQUIRE_EFFICIENCY_JSON = True` and point `EFFICIENCY_JSON_PATH` at the
matching `_efficiency.json` file for the selected model/backend/method pool.
If `EFFICIENCY_JSON_PATH` is left blank, the notebook searches the repo outputs
and selects the most recent matching file for the selected dataset, model, and
backend.

The Colab bootloader supports both common Drive locations:

```text
/content/drive/Othercomputers/.../activation-based-pruning
/content/drive/MyDrive/.../activation-based-pruning
/content/drive/Shared with me/.../activation-based-pruning
```

It also checks the historical misspelling `activation-based-prunning`, because
some shared Drive copies may use that folder name.

## Hybrid Demo Outputs

When `RUN_PRUNING_DEMO = True`, the adaptive hybrid notebook now writes:

```text
adaptive_hybrid_demo_summary.json
method_score_timing.csv
method_efficiency_by_ratio.csv
layer_method_score_stats.csv
layer_method_pair_agreement.csv
layer_decisions.csv
layer_decisions.json
adaptive_hybrid_layer_pruning_audit_<scope>_r<ratio>_c<threshold>.csv
adaptive_hybrid_layer_pruning_audit_<scope>_r<ratio>_c<threshold>.json
adaptive_hybrid_vs_methods_<scope>_r<ratio>_c<threshold>.csv
adaptive_hybrid_vs_methods_<scope>_r<ratio>_c<threshold>.json
adaptive_hybrid_<scope>_<model>_<dataset>_r<ratio>_c<threshold>_raw.pth
adaptive_hybrid_<scope>_<model>_<dataset>_r<ratio>_c<threshold>_healed.pth
```

The comparison CSV/JSON contains the selected hybrid stack alongside the
individual method-matrix rows at the same pruning ratio. This is the artifact to
use when comparing baseline accuracy, healed accuracy, FLOPs reduction,
parameter reduction, pruning time, healing time, and overall simplicity time.

The layer-level files are the transparency layer for the thesis:

- `method_score_timing.csv` records how long each candidate method took to
  score the model under the calibration-batch setting.
- `layer_method_score_stats.csv` records score distribution statistics and
  amortized scoring time per method/layer.
- `layer_method_pair_agreement.csv` records pairwise Spearman agreement and
  prune-set overlap for every layer, ratio, and threshold.
- `layer_decisions.csv` records the selected method or stack per layer, the
  proxy choices, the measured utility values, and whether ratio-matched
  efficiency evidence was available.
- `adaptive_hybrid_layer_pruning_audit_*.csv` records the final structural mask
  per layer: channels kept, channels pruned, keep ratio, selected stack, and
  decision rationale.

The notebook's final report-visuals section then converts these tables into
figures for the thesis and presentation:

- hybrid versus singular methods for healed accuracy, accuracy change, FLOPs
  reduction, parameter reduction, and simplicity time;
- Pareto scatter plots showing accuracy retention versus FLOPs reduction, with
  marker size representing pruning plus healing time;
- ratio/threshold sensitivity heatmaps;
- layer-wise method-pair agreement heatmaps;
- selected stack size and final pruned-channel ratio per layer.

## Efficiency Ranking From Experiments

The method-matrix notebooks rank methods with this priority:

```text
1. status == ok
2. lower simplicity_time_sec
3. higher healed_accuracy_delta_pct
4. higher healed_flops_reduction_pct
5. higher healed_params_reduction_pct
```

This matches the research rule:

```text
Prefer the method that is faster to prune and heal. If methods have similar
time, prefer the one that preserves accuracy better. If accuracy is comparable,
prefer the one with higher FLOPs reduction.
```

Inside the adaptive hybrid notebook, the same evidence is used in a continuous
utility score. This avoids treating two methods with very different FLOPs
benefits as equivalent simply because they have similar pruning time.

## Thesis Framing

Suggested name:

```text
Correlation- and Cost-Aware Adaptive Hybrid Pruning
```

The research claim:

```text
The proposed hybrid method does not combine criteria uniformly. Instead, it
measures layer-wise agreement between pruning criteria using Spearman rank
correlation and prune-set overlap. It replaces expensive criteria with cheaper
proxies whenever their channel rankings and selected prune sets are sufficiently
similar. When methods disagree, it uses a cost-aware weighted fusion of
complementary criteria.
```

## Literature Positioning

This strategy is supported by three literature directions:

- pruning criteria can produce highly similar channel rankings in some layers;
- different criteria have different computational costs;
- adaptive or learned criterion selection can outperform a fixed criterion
  applied from the first layer to the last.

Key papers to cite:

- Rethinking the Pruning Criteria for Convolutional Neural Network.
- Learning Filter Pruning Criteria for Deep Convolutional Neural Networks
  Acceleration.
- Loss-Aware Automatic Selection of Structured Pruning Criteria for Deep Neural
  Network Acceleration.
