# Experiment Metrics Schema

The `experiments_custom_method_registration_minimal*.ipynb` notebooks save two
files after the full method matrix runs:

```text
outputs/custom_method_matrix/notebook_run_<model>_<preset>_<timestamp>.json
outputs/custom_method_matrix/notebook_run_<model>_<preset>_<timestamp>_efficiency.json
```

The efficiency JSON is the main input for adaptive hybrid pruning. For the
thesis experiment, generate this ranking file before running the adaptive hybrid
notebook so "simple" is based on measured method cost plus healing time, not on
hand-written fallback costs.

The method matrix intentionally includes both bundled methods and registered
custom methods. At minimum the ranking file must contain successful rows for
the methods used by the hybrid notebook, including:

```text
l1_norm, mean_abs_act, apoz, custom_l2, chip, custom_reprune, custom_tis, custom_nisp
```

If the hybrid notebook reports missing measured simplicity costs, rerun the
matching `experiments_custom_method_registration_minimal*.ipynb` notebook after
confirming that `METHOD_MATRIX_METHODS` contains those methods.

## Core Identity Fields

- `backend`: `pytorch` or `keras`
- `dataset`: dataset key, for example `cifar-10`
- `model`: model name, for example `resnet18`
- `method`: pruning method name
- `scope`: `local` or `global`
- `prune_ratio`: pruning ratio used in the run
- `status`: `ok` or `error`
- `error`: error message if the run failed

## Timing Fields

- `prune_time_sec`: overall method pruning cost reported by the pruning engine.
  In current runs this is `score_time_sec + mask_build_time_sec +
  surgery_time_sec`.
- `score_time_sec`: time spent computing the method's channel-importance scores.
- `mask_build_time_sec`: time spent converting scores to pruning masks.
- `surgery_time_sec`: time spent physically applying the masks to create the
  pruned model.
- `method_cost_time_sec`: same cost basis as `prune_time_sec`; included to make
  the ranking basis explicit in new outputs.
- `prune_pipeline_time_sec`: full pruning pipeline wall time, including topology
  analysis and checkpoint save overhead when present.
- `heal_time_sec`: time spent fine-tuning/recovering the pruned model.
- `simplicity_time_sec`: `prune_time_sec + heal_time_sec`.
- `wall_time_sec`: full method-loop wall time, including evaluation and saving.

For the hybrid method, simplicity is primarily defined as:

```text
simplicity_time_sec = prune_time_sec + heal_time_sec
```

## Baseline Fields

- `baseline_accuracy_pct`
- `baseline_flops`
- `baseline_params`
- `baseline_ckpt_used`
- `baseline_created`

## Forcing Baseline Retraining

Each `experiments_custom_method_registration_minimal*.ipynb` notebook exposes:

```python
FORCE_RETRAIN_BASELINE = False
```

Set it to `True` when you want the notebook to ignore any existing baseline
checkpoint and train a fresh baseline for the current model/backend/run. The new
checkpoint is saved under:

```text
saved_models/baselines/<backend>/<dataset>/<model>/
```

Use this when changing the dataset, training preset, baseline epochs, or when
you do not trust an older checkpoint. Leave it as `False` when you want to reuse
the latest matching baseline checkpoint and go straight to the method matrix.

## Raw Pruned Fields

These are measured immediately after structural pruning and before healing:

- `raw_pruned_accuracy_pct`
- `raw_pruned_accuracy_delta_pct`
- `raw_pruned_flops`
- `raw_pruned_params`
- `raw_pruned_flops_reduction_pct`
- `raw_pruned_params_reduction_pct`
- `raw_pruned_ckpt`

## Healed Fields

These are measured after fine-tuning/recovery:

- `healed_accuracy_pct`
- `healed_accuracy_delta_pct`
- `healed_flops`
- `healed_params`
- `healed_flops_reduction_pct`
- `healed_params_reduction_pct`
- `healed_ckpt`

## Efficiency Ranking Fields

The `_efficiency.json` file adds:

- `efficiency_rank`
- `efficiency_rank_basis`
- `efficiency_accuracy_delta_pct`
- `efficiency_flops_reduction_pct`
- `efficiency_params_reduction_pct`
- `simplicity_time_bucket_sec`

Ranking priority:

```text
1. lower simplicity_time_sec
2. higher healed_accuracy_delta_pct
3. higher healed_flops_reduction_pct
4. higher healed_params_reduction_pct
```

## Using Metrics in Hybrid Pruning

Pass the efficiency JSON path into the hybrid config:

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

ReduCNN then uses measured method-matrix rows when deciding whether a cheaper
method can proxy for a more expensive method in each layer. The adaptive
notebook prefers rows measured at the same `prune_ratio` as the hybrid run. If
no same-ratio row exists, and cross-ratio fallback is enabled, the notebook uses
the nearest available ratio and records this in the layer-decision outputs.

The layer selector is not based on time alone. It computes a measured utility
from:

- `simplicity_time_sec`
- `healed_accuracy_delta_pct`
- `healed_flops_reduction_pct`
- `healed_params_reduction_pct`

This means a fast method is not automatically selected if it gives weak FLOPs
reduction or poor healed accuracy.

In the dataset-agnostic notebook, set:

```python
REQUIRE_EFFICIENCY_JSON = True
EFFICIENCY_JSON_PATH = "outputs/custom_method_matrix/notebook_run_resnet18_balanced_YYYYMMDD_HHMMSS_efficiency.json"
```

The notebook will stop if the file is missing or if the selected method pool has
methods without measured costs. Set `REQUIRE_EFFICIENCY_JSON = False` only for a
quick smoke test.

The adaptive notebook also writes a data-backed ranking table before doing the
hybrid sweep:

```text
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/method_efficiency_ranking.csv
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/method_efficiency_ranking.json
```

This table is filtered to the selected dataset, model, and backend. It exposes
raw median `prune_time_sec`, `heal_time_sec`, `simplicity_time_sec`, separate
time ranks, and the normalized `relative_simplicity_cost` used by the hybrid
selector. Use this file to verify claims such as "CHIP is expensive" against the
actual run data.

`SCORING_CALIB_BATCHES` in the adaptive notebook is a scoring/calibration pass
limit, not an epoch count. It controls how many batches are used to compute
method score maps for layer-wise correlation. The training/healing epoch count
is still controlled separately by the experiment preset's `finetune_epochs`.
For Colab stability, the adaptive notebook also allows method-specific scoring
limits through `METHOD_SCORING_BATCHES`; CHIP defaults to one scoring batch on
VGG-style runs because its activation/SVD scoring path can be memory-heavy.

## Adaptive Hybrid Dataset-Agnostic Notebook

The `experiments_adaptive_hybrid_dataset_agnostic.ipynb` notebook writes:

```text
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/method_efficiency_ranking.csv
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/method_efficiency_ranking.json
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/method_efficiency_by_ratio.csv
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/method_efficiency_by_ratio.json
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/method_score_timing.csv
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/method_score_timing.json
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/layer_method_score_stats.csv
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/layer_method_score_stats.json
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/layer_method_pair_agreement.csv
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/layer_method_pair_agreement.json
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/layer_decisions.csv
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/layer_decisions.json
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/band_summary.csv
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/stack_size_threshold_sweep.png
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/adaptive_hybrid_demo_summary.json
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/adaptive_hybrid_demo_matrix.csv
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/adaptive_hybrid_demo_matrix.json
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/adaptive_hybrid_layer_pruning_audit_<scope>_r<ratio>_c<threshold>.csv
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/adaptive_hybrid_layer_pruning_audit_<scope>_r<ratio>_c<threshold>.json
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/adaptive_hybrid_vs_methods_<scope>_r<ratio>_c<threshold>.csv
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/adaptive_hybrid_vs_methods_<scope>_r<ratio>_c<threshold>.json
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/adaptive_hybrid_stack_effectiveness_<scope>_r<ratio>_c<threshold>.csv
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/adaptive_hybrid_stack_effectiveness_<scope>_r<ratio>_c<threshold>.json
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/adaptive_hybrid_stack_effectiveness_summary_<scope>_r<ratio>_c<threshold>.json
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/adaptive_hybrid_<scope>_<model>_<dataset>_r<ratio>_c<threshold>_raw.pth
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/adaptive_hybrid_<scope>_<model>_<dataset>_r<ratio>_c<threshold>_healed.pth
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/report_visuals/*.png
outputs/adaptive_hybrid/<dataset>/<model>/<timestamp>/report_visuals/report_visuals_manifest.json
```

The stack-effectiveness files answer "did the selected stack work?" directly.
They compare the adaptive hybrid against the individual methods that appeared in
the selected layer-wise stack and report:

- accuracy delta versus each stack member;
- baseline accuracy delta gain versus each stack member;
- FLOPs and parameter reduction gains;
- pruning, healing, and total simplicity time saved;
- boolean win/loss flags for accuracy retention, FLOPs, params, pruning time,
  healing time, and total simplicity time;
- an overall `stack_worked_overall` flag in the summary JSON.

The demo matrix files contain one hybrid pruning/healing run per requested
scope, pruning ratio, and correlation threshold. This is the main artifact for
seeing whether the hybrid behavior is stable across experimental settings. If
baseline accuracy is below `MIN_BASELINE_ACC_FOR_INTERPRETATION`, the matrix
sets `baseline_interpretable = false`; timing and compression metrics can still
be inspected, but accuracy-retention claims should not be used.

The layer-decision files include:

- `ratio`
- `correlation_threshold`
- `overlap_threshold`
- `layer`
- `band`
- `candidate_methods`
- `simple_candidates`
- `complex_candidates`
- `mode`
- `selected`
- `stack`
- `stack_size`
- `weights`
- `covered_complex`
- `proxy_choices`
- `selection_reason`
- `method_efficiency`
- `efficiency_evidence_modes`
- `has_ratio_matched_efficiency`
- `uses_cross_ratio_efficiency`
- `max_abs_pair_correlation`
- `max_pair_overlap`

The pair-agreement file is the main source for plots such as "for VGG16
features.10, which methods agreed at 55% or 80% threshold?" It includes:

- `method_a`
- `method_b`
- `spearman_rank_corr`
- `prune_set_overlap`
- `passes_correlation_threshold`
- `passes_overlap_threshold`
- method utility and FLOPs-reduction columns for both methods.

The layer-pruning audit file is the final explanation of what actually happened
after mask construction. It includes:

- `kept_channels`
- `pruned_channels`
- `keep_ratio`
- `pruned_ratio`
- `selected`
- `stack`
- `weights`
- `selection_reason`

The final report-visuals section reads these artifacts and saves presentation
figures without recomputing pruning. The main figures are:

- `hybrid_vs_methods_<tag>_healed_accuracy_pct.png`
- `hybrid_vs_methods_<tag>_healed_accuracy_delta_pct.png`
- `hybrid_vs_methods_<tag>_healed_flops_reduction_pct.png`
- `hybrid_vs_methods_<tag>_healed_params_reduction_pct.png`
- `hybrid_vs_methods_<tag>_simplicity_time_sec.png`
- `pareto_accuracy_flops_time_<tag>.png`
- `hybrid_sensitivity_<scope>_<metric>.png`
- `layer_pair_abs_spearman_rank_corr_<scope>_r<ratio>_c<threshold>.png`
- `layer_pair_prune_set_overlap_<scope>_r<ratio>_c<threshold>.png`
- `selected_stack_size_by_layer.png`
- `layer_pruned_ratio_<scope>_r<ratio>_c<threshold>.png`

These plots are designed to support the thesis claim that the adaptive hybrid is
chosen by layer-wise agreement and measured efficiency, then evaluated against
the individual methods that appear in the stack.

The notebook sweeps correlation thresholds from permissive to strict:

```text
0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
```

It also enforces the rule that simple-only method comparisons do not create a
hybrid stack. Simple methods can proxy complex methods, or one simple method can
appear as a representative in a mixed stack.
