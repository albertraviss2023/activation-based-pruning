# Experiment Metadata Registry

The LFPC notebooks now produce enough artifacts that file discovery should be handled by a registry instead of ad hoc path searches.

`scripts/build_experiment_registry.py` scans `outputs/lfpc_hybrid` and writes a normalized metadata layer under `reports/experiment_registry`.

The registry is not the final top-stack analysis. It is the first step a reporting notebook should run so the notebook knows which time-stamped runs produced which contexts and artifacts. Final ranking, filtering, and thesis-facing claims still belong in the analysis notebook.

## Build

```bash
python scripts/build_experiment_registry.py
```

Useful options:

```bash
python scripts/build_experiment_registry.py ^
  --outputs-root outputs/lfpc_hybrid ^
  --registry-dir reports/experiment_registry ^
  --accuracy-gate-pp 7 ^
  --top-k 5
```

## Registry Files

- `registry_manifest.json`: schema version, generation time, counts, and file names.
- `runs.csv`: one row per detected experiment run directory.
- `artifacts.csv`: one row per artifact file, including artifact role, path, size, and modified time.
- `contexts.csv`: normalized benchmark rows for hybrid stacks and singular methods.
- `context_run_index.csv`: one row per run-specific context, preserving timestamp and similarity settings.
- `latest_context_runs.csv`: latest run for each exact context key.
- `singular_cache_index.csv`: latest reusable singular benchmark/checkpoint per dataset × model × scope × ratio × method.
- `context_summary.csv`: one row per objective × dataset × model × scope × prune ratio, with metric coverage and best values.
- `best_hybrid_by_context.csv`: convenience table only; final top-stack logic should be owned by the analysis notebook.
- `registry_quality_audit.csv`: warnings about missing metrics, missing checkpoint paths, or incomplete context keys.
- `*.jsonl`: line-delimited mirrors of the main CSVs for streaming or non-pandas consumers.

## Core Context Fields

Reporting notebooks should join and filter by these fields before comparing artifacts:

- `dataset`
- `model`
- `objective`
- `scope`
- `ratio`
- `variance_threshold`
- `spearman_threshold`
- `jaccard_threshold`
- `record_type`
- `stack_id`
- `method`

For comparisons, never mix different values of `dataset`, `model`, `scope`, or `ratio`.

## Query Examples

Recommended first cell in an analysis notebook:

```python
import subprocess
import sys
from pathlib import Path

subprocess.run(
    [
        sys.executable,
        "scripts/build_experiment_registry.py",
        "--outputs-root", "outputs/lfpc_hybrid",
        "--registry-dir", "reports/experiment_registry",
    ],
    check=True,
)
```

```python
from pathlib import Path
import pandas as pd

registry = Path("reports/experiment_registry")
context_runs = pd.read_csv(registry / "context_run_index.csv")
latest = pd.read_csv(registry / "latest_context_runs.csv")
contexts = pd.read_csv(registry / "contexts.csv")

latest.head()
```

Find contexts that are well-populated enough for final reporting:

```python
summary = pd.read_csv(registry / "context_summary.csv")
ready = summary[
    (summary["hybrid_rows"] > 0)
    & (summary["singular_rows"] > 0)
    & (summary["hybrid_checkpoint_coverage"] > 0)
]
```

Find all artifacts for one top stack:

```python
artifacts = pd.read_csv(registry / "artifacts.csv")

row = latest.iloc[0]
run_artifacts = artifacts[artifacts["run_id"] == row["run_id"]]
run_contexts = contexts[
    (contexts["run_id"] == row["run_id"])
    & (contexts["dataset"] == row["dataset"])
    & (contexts["model"] == row["model"])
    & (contexts["scope"] == row["scope"])
    & (contexts["ratio"] == row["ratio"])
]
```

Find same-context singular benchmarks for a hybrid stack:

```python
row = latest.iloc[0]
singular_cache = pd.read_csv(registry / "singular_cache_index.csv")
singular = singular_cache[
    (singular_cache["dataset"] == row["dataset"])
    & (singular_cache["model"] == row["model"])
    & (singular_cache["scope"] == row["scope"])
    & (singular_cache["ratio"] == row["ratio"])
]
```

The singular cache may come from an earlier run than the hybrid stack. That is expected. The key rule is that `dataset`, `model`, `scope`, `ratio`, and `method` must match exactly, and the cache source run remains visible through `cache_source_run_id`.

## Future Notebook Contract

Every experiment notebook should write a small `run_manifest.json` and `run_manifest.txt` into `OUT_DIR` immediately after `OUT_DIR` is created. The registry uses these files first, then falls back to CSV/path inference for older runs.

Recommended manifest fields:

```json
{
  "schema_version": "2026-05-20.1",
  "run_id": "lfpc_CIFAR-10_VGG16_flops_accuracy_20260520_075543",
  "dataset": "CIFAR-10",
  "model": "VGG16",
  "objective": "flops_accuracy",
  "objective_label": "FLOPs + Accuracy",
  "prune_ratios": [0.3, 0.45, 0.55],
  "scopes": ["local", "global"],
  "similarity_grid": {
    "variance_thresholds": [0.05, 0.1, 0.2],
    "spearman_thresholds": [0.5, 0.7],
    "jaccard_thresholds": [0.5, 0.7]
  },
  "artifacts": {
    "hybrid_benchmarks": "fixed_hybrid_stack_benchmarks.csv",
    "singular_benchmarks": "current_run_singular_method_benchmarks.csv",
    "layer_policy": "lfpc_discovered_layer_policy_phase1.csv"
  }
}
```

This keeps reporting notebooks focused on analysis instead of filesystem archaeology.
