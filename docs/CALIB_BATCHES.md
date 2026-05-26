# Calibration Batches in ReduCNN

## Short Answer

`calib_batches` is an engineering speed knob, not a universally
literature-backed constant. It is acceptable for smoke tests and iteration, but
using very small calibration batches for final thesis results can compromise
validity unless we show that the pruning rankings and masks are stable.

## What `calib_batches` Means

In ReduCNN, `calib_batches` limits how much data activation-based methods see
when computing filter or channel importance scores.

It affects methods such as:

```text
mean_abs_act
APoZ
CHIP
entropy
class entropy
HRank
TIS
SENPIS
ThiNet-style
RePrune-style
```

It matters less for pure weight-based methods such as:

```text
L1
L2
```

because those methods do not need calibration activations.

## Is a Small `calib_batches` Value Backed by Literature?

Not as a fixed number such as "use 1 batch" or "use 2 batches."

The literature usually does one of the following:

- uses the training or validation set, or a subset, to estimate activation or
  filter importance;
- samples images or feature-map locations to reduce cost;
- reports final accuracy, FLOPs, parameters, or inference speed;
- sometimes includes ablations to justify that a smaller sample is enough.

For example:

- APoZ and Network Trimming are data-driven because they use activation
  statistics, so the representativeness of calibration data matters.
- ThiNet uses sampled data or sampled feature-map locations for filter selection
  because using everything would be expensive.
- HRank argues that average feature-map rank is relatively stable across
  different numbers of image batches. This supports smaller calibration budgets
  for HRank-like ranking, but it does not automatically validate small
  calibration budgets for APoZ, entropy, CHIP, TIS, SENPIS, or ThiNet.

## Are We Compromising Results?

For smoke tests, no. Small calibration settings are useful for checking whether
the pipeline runs.

For thesis experiments, yes, if we use tiny calibration settings without
validating stability.

The presets should be interpreted as:

```text
quick    = debugging / smoke test
balanced = exploratory experiments
full     = thesis/reportable candidate
```

The `quick` preset is not intended to support final claims. It is intended to
confirm that the pipeline runs.

## How Speed Is Handled in the Literature

Speed is commonly handled in four ways.

### 1. Sampling During Scoring

Instead of using the full dataset, pruning methods often sample images, batches,
or spatial locations.

### 2. Reporting Compression Metrics

Common reported metrics include:

```text
FLOPs reduction
parameter reduction
inference speedup
accuracy drop
```

### 3. Fine-Tuning After Pruning

Many pruning papers prune and then fine-tune or retrain. Recovery cost is often
accepted as part of the method, although it is not always compared fairly.

### 4. Reducing or Avoiding Retraining Cost

Some methods explicitly reduce recovery time, for example by pruning while
training or by designing cheaper criteria.

## Recommended ReduCNN Methodology

We should not claim that `calib_batches=1` or `calib_batches=2` is
literature-backed.

Instead, calibration size should be treated as an experimental stability
variable:

```text
calib_batches = 1, 2, 4, 8, 16
```

For each setting, measure:

```text
score rank stability
mask overlap stability
final accuracy
FLOPs reduction
parameter reduction
prune time
heal time
```

A defensible thesis statement is:

```text
Calibration size was treated as an experimental stability variable. Final
reported pruning results use the smallest calibration setting whose filter
rankings and pruning masks remain stable relative to larger calibration
settings.
```

## Recommendation for Final Experiments

- Use `quick` only for debugging.
- Use `balanced` for preliminary comparisons.
- Use `full` or a validated calibration size for thesis tables.
- Add a calibration ablation for at least one or two models:

```text
calib_batches: 1, 2, 4, 8, 16
```

- Report when rankings stabilize using Spearman correlation and prune-set
  overlap.

This lets us say that we are not choosing small batches blindly for speed. We are
empirically validating the smallest reliable calibration budget.
