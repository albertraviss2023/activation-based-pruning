# Cross-Backend Red-Flag Review (DenseNet121 Custom Methods)

Date: 2026-03-24
Context: `experiments_custom_method_registration_*` matrix output (Keras vs PyTorch, same methods/model)

## Snapshot verdict
- Overall execution health: good (`status=ok` for all methods on both backends).
- Structural comparability: good (`layers_scored=120` on both backends for all methods).
- Primary future-risk areas: runtime asymmetry on data-heavy methods, baseline-source mismatch, and very low per-layer minima in global pruning.

## What looks healthy (not a red flag)
1. Local methods show near-identical pruning behavior across backends.
- `custom_l2`, `custom_entropy`, `custom_hrank`, `custom_class_entropy`, `custom_spectral_energy`
- `mean_keep_ratio` and `min_keep_ratio` match almost exactly (around `0.6955 / 0.6875`).
- Interpretation: mask construction logic and local-threshold semantics are consistent cross-backend.

2. Global methods are directionally aligned.
- `chip`, `custom_nisp`, `custom_reprune`, `custom_senpis`, `custom_thinet`, `custom_tis`
- Keep ratios are in similar ranges between backends (small expected drift).
- Interpretation: same high-level pruning intent is preserved.

## Red flags to track (future work)
1. Baseline provenance mismatch (major confounder).
- Keras rows: `baseline_created=True`
- PyTorch rows: `baseline_created=False`
- Why this matters: backend comparison can be polluted by different baseline quality/training histories.
- Priority: high for scientific fairness.

2. Extreme runtime gap for data-intensive methods in Keras.
- Examples (`wall_time_sec`):
  - `custom_senpis`: Keras ~239s vs PyTorch ~53s
  - `custom_tis`: Keras ~237s vs PyTorch ~52s
  - `chip`: Keras ~76s vs PyTorch ~24s
- Why this matters: not a correctness bug by itself, but practical usability and reproducibility risk.
- Likely cause: Keras-side activation/gradient probing overhead.
- Priority: medium-high.

3. Very low `min_keep_ratio` in global methods (both backends, but worth guarding).
- `chip`/`custom_nisp`: minima near `0.0020`
- Interpretation: some layers are almost completely collapsed.
- Why this matters: high instability risk on other seeds/models, can hurt recovery sharply.
- Priority: medium.

4. `prune_time_sec` vs `wall_time_sec` spread is very large.
- Surgery is fast; score computation dominates wall time.
- Not a bug, but can mislead users if they interpret `prune_time_sec` as end-to-end method cost.
- Priority: medium (UX/metrics clarity).

## Non-red-flag differences (expected)
1. Small keep-ratio drifts on global methods.
- Caused by floating-point/tie ordering differences and framework internals.

2. Keras generally slower than PyTorch in this setup.
- Expected for this type of per-layer probing pipeline (especially with repeated model calls).

## Recommended guardrails for future experiments
1. Enforce baseline parity before cross-backend claims.
- Same architecture family and comparable baseline quality threshold.
- Block comparison if one side is auto-created weak baseline.

2. Add collapse guard for global pruning.
- Layer minimum keep floor (e.g., per-layer lower bound) to avoid pathological near-zero layers.

3. Report both timing metrics explicitly.
- Keep `prune_time_sec` (surgery only) and `wall_time_sec` (end-to-end), and label them clearly in plots/tables.

4. Add method-specific runtime notes in notebook output.
- Flag `chip`, `senpis`, `tis` as data/gradient-heavy and expected slower classes.

## Suggested severity ranking
- High: baseline provenance mismatch (`baseline_created` differs across backends).
- Medium-high: Keras runtime inflation for heavy methods.
- Medium: global minimum keep collapse risk.
- Medium: timing metric interpretation ambiguity.

## Bottom line
- There is no immediate sign of cross-backend correctness collapse in the DenseNet121 matrix.
- The biggest comparability risk is baseline mismatch, not mask logic.
- The biggest practical risk is Keras runtime for gradient-heavy methods.
