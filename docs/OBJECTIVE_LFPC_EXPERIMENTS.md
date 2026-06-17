# Objective LFPC Experiments

This document explains the objective-driven LFPC experiment notebooks used for
the thesis. It focuses on the contract each notebook should satisfy so later
analysis can compare hybrid stacks and singular methods without mixing contexts.

## Purpose

The objective LFPC notebooks search for layerwise hybrid pruning policies. A
policy assigns one pruning method to each prunable layer, then benchmarks the
resulting fixed stack after structural pruning and healing.

The notebooks answer three linked questions:

1. Which hybrid stacks are strongest for each optimization objective?
2. How do those stacks differ by dataset, architecture, scope, and pruning ratio?
3. Which methods are selected in early, middle, and late layers, and what
   accuracy, FLOPs, parameter, and pruning-time trade-off do they achieve?

## Notebook Naming Convention

The experiment notebooks follow this pattern:

```text
experiments_for_pruning_policy_search_on_context_<dataset>_<model>_registered_methods_<objective>.ipynb
```

Common axes:

- Dataset: `cifar10`, `cats_dogs`
- Model: `vgg16`, `resnet18`
- Objective: `objective_flops_accuracy`, `objective_time_accuracy`, `all_three`
- Scope: `local`, `global`
- Pruning ratio: usually values such as `0.30`, `0.45`, `0.55`
- Similarity thresholds: variance, Spearman, and Jaccard settings used by the
  policy-search candidate filtering stage

## Objectives

Each objective ranks stacks differently and should be reported separately.

- **FLOPs + Accuracy**: favor high test-accuracy retention and high structural
  FLOPs reduction.
- **Time + Accuracy**: favor high test-accuracy retention and low fixed-stack
  pruning time.
- **FLOPs + Time + Accuracy**: balance test-accuracy retention, structural FLOPs
  reduction, and pruning time.

The exact ranking weights can evolve in notebooks, but reporting must always
state the objective and must not rank stacks globally across incompatible
contexts.

## Runtime Definition

The fixed-stack benchmark runtime is intended to measure the cost of applying a
selected pruning policy, not the cost of discovering it. Notebook exports should
exclude LFPC policy-search overhead and capture the fixed pruning workflow:

```text
method scoring + mask construction + structural pruning + healing/evaluation
```

If a notebook uses a narrower timing definition, record that in the run metadata
and exported timing-source columns.

## Required Context Keys

Every exported artifact used for reporting should carry these keys:

- `objective_label` or objective scenario
- `dataset`
- `model`
- `scope`
- `ratio`
- `variance_threshold`
- `spearman_threshold`
- `jaccard_threshold`
- run identifier and timestamp

For hybrid stacks, also include:

- `stack_id`
- stable short/report stack id
- selected method per layer
- layer name and layer index
- layer region, when available: early, middle, late

For singular methods, also include:

- `method`
- `method_display`
- checkpoint path
- cache source or run source

## Core Artifact Contract

Each notebook should save these artifacts under its run directory:

- `run_metadata.json`: dataset, model, objective, ratios, thresholds, flags,
  timestamp, git/package information when available.
- `method_score_timing.csv`: scoring time and calibration budget per method.
- `lfpc_discovered_layer_policy_phase1.csv`: layerwise method decisions from
  policy discovery.
- `fixed_hybrid_stack_benchmarks.csv`: one row per benchmarked hybrid stack with
  accuracy, FLOPs, parameters, time, and checkpoint provenance.
- `current_run_singular_method_benchmarks.csv`: singular method benchmark rows
  for the same dataset, model, scope, and ratio contexts.
- `artifact_completeness_audit.csv`: schema, metric, checkpoint, and scope
  checks.
- `top_stack_reporting/`: notebook-local plots and tables for top hybrid stacks
  compared against same-context singular methods.
- `phase2_phase3_outputs/`: standardized benchmark and stability diagnostics,
  when the notebook runs the Phase 2/3 reporting cell.

## Model Checkpoint Contract

Hybrid and singular pruned models should be saved with enough metadata to
reconstruct their context:

- dataset
- model
- objective, for hybrid stacks
- scope
- ratio
- stack id or singular method
- timestamp/run id
- baseline checkpoint used
- final accuracy and compression metrics, when available

Singular benchmarks are reusable across objectives when the dataset, model,
scope, and pruning ratio match exactly. If a required singular benchmark is
missing for a context, the experiment workflow should run that missing singular
case, save the checkpoint, and export the benchmark row for future reuse.

## Comparability Rules

The reporting notebooks should enforce these rules:

1. Never compare different datasets.
2. Never compare different architectures.
3. Never compare local and global scopes.
4. Never compare different pruning ratios.
5. Do not substitute singular rows from another objective unless the dataset,
   model, scope, and ratio match and the source is clearly labeled.
6. Use structural FLOPs and parameter reductions measured from the pruned model,
   not just requested pruning ratio.
7. Keep hybrid stack identifiers stable across layerwise plots, Pareto plots,
   comparison plots, and tables.

## Important Configuration Knobs

Common notebook flags:

- `FORCE_RETRAIN_BASELINE`: retrain the baseline even if a checkpoint exists.
- `FORCE_REPRUNE_FIXED_STACKS`: recompute hybrid pruned models even when cached.
- `REPRUNE_SINGULAR_METHODS`: recompute all singular benchmarks instead of
  loading saved ones.
- `RUN_MISSING_SINGULAR_BENCHMARKS`: when singular caching is enabled, run and
  save only missing dataset-model-scope-ratio-method contexts.
- `TOP_STACK_REPORT_K`: number of ranked hybrid stacks to report per exact
  context.
- `MAX_ALLOWED_ACCURACY_DROP_PCT`: accuracy gate used for reporting and ranking.
- `LFPC_TARGET_FLOPS_REDUCTION`: optional objective target for FLOPs-aware
  ranking.

## Reporting Expectations

For each exact context:

```text
objective x dataset x model x scope x pruning ratio
```

the report should show:

- top hybrid stacks, normally the top two or top four depending on the thesis
  section;
- layerwise policy plot for each top stack;
- a table directly under the plot listing layer, region, method, and layer
  pruning ratio;
- hybrid vs same-context singular method comparison for accuracy delta, FLOPs
  reduction, parameter reduction where available, and pruning time;
- Pareto view with stable hybrid stack ids and singular method labels.

## Troubleshooting Checklist

Use this checklist before trusting a report:

- Are any required metric columns missing or all-NaN?
- Do all singular rows match the hybrid row on dataset, model, scope, and ratio?
- Are hybrid and singular checkpoints present?
- Are local/global methods restricted to their allowed scope?
- Does the plotted stack id match the layerwise policy table?
- Does the runtime source match the intended benchmark definition?
- Are cached singular benchmarks labeled with their source run?

If any check fails, fix the experiment artifact first rather than patching the
analysis notebook to hide the inconsistency.
