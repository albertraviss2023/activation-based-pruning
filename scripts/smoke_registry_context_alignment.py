"""Smoke test registry context alignment and reusable singular cache lookup.

This is intentionally light: it does not run pruning. It verifies that a
notebook-style manifest can be written, cached singular benchmarks can be found
by exact context, and analysis joins do not mix dataset/model/scope/ratio.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, default=Path("reports/experiment_registry"))
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("experiments_for_pruning_policy_search_on_context_vgg16_cifar10_registered_methods_objective_time_accuracy.ipynb"),
    )
    return parser.parse_args()


def run_notebook_manifest_smoke(notebook: Path, registry_dir: Path) -> dict[str, int]:
    marker = "# --- LFPC run manifest metadata ---"
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    manifest_cell = next("".join(c.get("source", [])) for c in nb["cells"] if marker in "".join(c.get("source", [])))
    compile(manifest_cell, str(notebook), "exec")

    tmp = Path(tempfile.mkdtemp(prefix="lfpc_manifest_smoke_"))
    globals_for_cell = {
        "OUT_DIR": tmp,
        "ROOT": Path.cwd(),
        "RUN_STAMP": "20990101_000000",
        "DATASET_KEY": "cifar10",
        "MODEL_TARGET": "vgg16",
        "BACKEND": "pytorch",
        "OBJECTIVE_SCENARIO": "time_accuracy",
        "OBJECTIVE_SCENARIO_LABEL": "Time + Accuracy",
        "OPTIMIZED_OBJECTIVE_TERMS": ("time", "accuracy"),
        "PRUNE_RATIOS": [0.30, 0.45, 0.55],
        "DEMO_PRUNE_RATIOS": [0.30, 0.45, 0.55],
        "SCORE_SCOPE_MODE": "all",
        "ALGORITHM2_VARIANCE_GRID": [0.05, 0.1, 0.2],
        "ALGORITHM2_SPEARMAN_GRID": [0.5, 0.7],
        "ALGORITHM2_JACCARD_GRID": [0.5, 0.7],
        "METHODS_BY_SCOPE": {"local": ["l1_norm", "custom_l2", "apoz"], "global": ["chip", "custom_nisp"]},
        "METHOD_CANDIDATES": ["l1_norm", "custom_l2", "apoz", "chip", "custom_nisp"],
        "META": {"dataset_label": "cifar-10"},
    }
    exec(compile(manifest_cell, str(notebook), "exec"), globals_for_cell)
    manifest = json.loads((tmp / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_key"] == "cifar10"
    assert manifest["model_target"] == "vgg16"
    assert manifest["objective_scenario"] == "time_accuracy"
    assert manifest["prune_ratios"] == [0.3, 0.45, 0.55]

    cache = pd.read_csv(registry_dir / "singular_cache_index.csv")
    sub = cache[
        (cache["dataset"] == "CIFAR-10")
        & (cache["model"] == "VGG16")
        & (cache["scope"].isin(["local", "global"]))
        & (cache["ratio"].isin([0.30, 0.45, 0.55]))
    ].copy()
    if sub.empty:
        raise RuntimeError("No cached singular rows found for VGG16/CIFAR-10 smoke context")
    return {
        "cached_singular_rows": int(len(sub)),
        "cached_singular_checkpoint_rows": int(sub["has_checkpoint_path"].fillna(False).astype(bool).sum()),
    }


def run_analysis_join_smoke(registry_dir: Path) -> dict[str, int]:
    latest = pd.read_csv(registry_dir / "latest_context_runs.csv")
    contexts = pd.read_csv(registry_dir / "contexts.csv")
    cache = pd.read_csv(registry_dir / "singular_cache_index.csv")
    artifacts = pd.read_csv(registry_dir / "artifacts.csv")

    hybrid_contexts = latest[latest["hybrid_rows"].fillna(0).astype(int) > 0].copy()
    key_cols = ["dataset", "model", "scope", "ratio"]
    joined = hybrid_contexts.merge(cache, on=key_cols, how="left", suffixes=("_hybrid_context", "_singular_cache"))
    matched = joined[joined["cache_key"].notna()].copy()
    missing = joined[joined["cache_key"].isna()].copy()
    if matched.empty:
        raise RuntimeError("No hybrid contexts matched the singular cache")

    cache_keys = set(cache["cache_key"].astype(str))
    unknown_keys = set(matched["cache_key"].dropna().astype(str)) - cache_keys
    if unknown_keys:
        raise RuntimeError(f"Joined rows reference unknown singular cache keys: {sorted(unknown_keys)[:5]}")

    checked_contexts = 0
    for _, row in hybrid_contexts.head(75).iterrows():
        ratio = float(row["ratio"]) if pd.notna(row["ratio"]) else np.nan
        ctx_rows = contexts[
            (contexts["run_id"].astype(str) == str(row["run_id"]))
            & (contexts["dataset"].astype(str) == str(row["dataset"]))
            & (contexts["model"].astype(str) == str(row["model"]))
            & (contexts["objective"].astype(str) == str(row["objective"]))
            & (contexts["scope"].astype(str) == str(row["scope"]))
            & (np.isclose(pd.to_numeric(contexts["ratio"], errors="coerce"), ratio, equal_nan=True))
        ].copy()
        if ctx_rows.empty:
            continue
        checked_contexts += 1
        bad = ctx_rows[
            (ctx_rows["dataset"].astype(str) != str(row["dataset"]))
            | (ctx_rows["model"].astype(str) != str(row["model"]))
            | (ctx_rows["scope"].astype(str) != str(row["scope"]))
            | (~np.isclose(pd.to_numeric(ctx_rows["ratio"], errors="coerce"), ratio, equal_nan=True))
        ]
        if not bad.empty:
            raise RuntimeError("Exact context extraction produced misaligned rows")

    return {
        "hybrid_contexts": int(len(hybrid_contexts)),
        "joined_rows": int(len(joined)),
        "matched_rows": int(len(matched)),
        "missing_join_rows": int(len(missing)),
        "checked_exact_context_extractions": int(checked_contexts),
        "checkpoint_artifacts": int((artifacts["artifact_role"].astype(str) == "model_checkpoint").sum()),
        "run_manifest_artifacts": int((artifacts["artifact_role"].astype(str) == "run_manifest").sum()),
    }


def main() -> None:
    args = parse_args()
    notebook_result = run_notebook_manifest_smoke(args.notebook, args.registry_dir)
    analysis_result = run_analysis_join_smoke(args.registry_dir)
    result = {
        "status": "ok",
        "notebook_manifest_smoke": notebook_result,
        "analysis_join_smoke": analysis_result,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
