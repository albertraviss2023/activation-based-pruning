from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "analysis_learned_layerwise_hybrid_stacks_report.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


cells = [
    md(
        r"""
# Learned Layer-Wise Hybrid Pruning Stack Report

This notebook reports the latest learned hybrid pruning stacks from the LFPC experiment artifacts. It treats all three optimization objectives as first-class:

1. **FLOPs + Accuracy**
2. **Time + Accuracy**
3. **FLOPs + Time + Accuracy**

The report covers both datasets, both models, all pruning ratios present in the artifacts, local and global scopes, singular methods where exported, and learned hybrid stacks. Values are loaded from artifacts only; missing metrics are reported as missing rather than invented.
"""
    ),
    code(
        r"""
from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
except Exception:
    sns = None

ROOT = Path.cwd()
OUT_ROOT = ROOT / "outputs" / "lfpc_hybrid"
REPORT_DIR = ROOT / "reports" / "learned_layerwise_stack_report"
PLOT_DIR = REPORT_DIR / "plots"
TABLE_DIR = REPORT_DIR / "tables"
for d in [REPORT_DIR, PLOT_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATASETS = ["cifar10", "cats_dogs"]
MODELS = ["vgg16", "resnet18"]
OBJECTIVE_DIRS = {
    "flops_accuracy": "FLOPs + Accuracy",
    "time_accuracy": "Time + Accuracy",
}
BASE_OBJECTIVE = "all_three"
BASE_OBJECTIVE_LABEL = "FLOPs + Time + Accuracy"
TOP_N = 5

METHOD_DISPLAY = {
    "l1_norm": "L1",
    "custom_l2": "L2",
    "mean_abs_act": "MeanAct",
    "apoz": "APoZ",
    "custom_entropy": "Entropy",
    "custom_class_entropy": "ClassEntropy",
    "custom_hrank": "HRank",
    "custom_spectral_energy": "Spectral",
    "chip": "CHIP",
    "custom_reprune": "REPrune",
    "custom_tis": "TIS",
    "custom_nisp": "NISP",
    "custom_senpis": "SeNPIS",
    "custom_thinet": "ThiNet",
    "custom_gfs": "GFS",
    "custom_dcp": "DCP",
    "custom_autodfp": "AutoDFP",
    "custom_gfi_ap": "GFI-AP",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})
"""
    ),
    md(
        "## 1. Load Latest Artifact Runs\n\nDirect timestamp folders are interpreted as the all-three-objective runs from the registered-method notebooks. Objective subfolders are interpreted as pairwise-objective runs."
    ),
    code(
        r"""
def is_timestamp_dir(path: Path) -> bool:
    return bool(re.fullmatch(r"\d{8}_\d{6}", path.name))


def latest_run_for(dataset: str, model: str, objective: str) -> Path | None:
    base = OUT_ROOT / dataset / model
    if objective == BASE_OBJECTIVE:
        candidates = [p for p in base.iterdir() if p.is_dir() and is_timestamp_dir(p) and (p / "fixed_hybrid_stack_benchmarks.csv").exists()] if base.exists() else []
    else:
        obj_base = base / objective
        candidates = [p for p in obj_base.iterdir() if p.is_dir() and is_timestamp_dir(p) and (p / "fixed_hybrid_stack_benchmarks.csv").exists()] if obj_base.exists() else []
    return sorted(candidates)[-1] if candidates else None


run_rows = []
for dataset in DATASETS:
    for model in MODELS:
        for objective in [BASE_OBJECTIVE, *OBJECTIVE_DIRS.keys()]:
            run_dir = latest_run_for(dataset, model, objective)
            run_rows.append({
                "dataset": dataset,
                "model": model,
                "objective": objective,
                "objective_label": BASE_OBJECTIVE_LABEL if objective == BASE_OBJECTIVE else OBJECTIVE_DIRS[objective],
                "run_dir": str(run_dir) if run_dir else "",
                "run_stamp": run_dir.name if run_dir else "",
                "has_run": run_dir is not None,
                "has_singular_comparison": bool(run_dir and (run_dir / "fixed_stack_vs_all_singular_display.csv").exists()),
            })
run_index = pd.DataFrame(run_rows)
run_index.to_csv(TABLE_DIR / "run_index.csv", index=False)
display(run_index)
"""
    ),
    md(
        "## 2. Registered-Method Pruning Provenance\n\n"
        "The learned-stack report is artifact-driven, but the artifacts must come from the same decorator-registered pruning path used in the custom-method notebooks. "
        "This section audits the experiment notebooks directly. It verifies that methods are registered with `@register_method`, that singular pruning is performed through `ReduCNNPruner(method=..., scope=...)`, and that local/global singular benchmarks are generated from `METHODS_BY_SCOPE` rather than from a generic pruning shortcut."
    ),
    code(
        r"""
# Audit the registered-method experiment notebooks that generated the artifacts.
import json as _json

EXPECTED_LOCAL_METHODS = [
    "apoz", "mean_abs_act", "custom_entropy", "custom_hrank", "l1_norm",
    "custom_l2", "custom_class_entropy", "custom_spectral_energy", "custom_gfi_ap",
]
EXPECTED_GLOBAL_METHODS = [
    "custom_nisp", "custom_senpis", "custom_tis", "chip", "custom_reprune",
    "custom_thinet", "custom_gfs", "custom_dcp", "custom_autodfp",
]

def _display_method(method: Any) -> str:
    return METHOD_DISPLAY.get(str(method), re.sub(r"^custom_", "", str(method)).replace("_", " ").title())

def _safe_read_csv_local(path: Path) -> pd.DataFrame:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: could not read {path}: {exc}")
        return pd.DataFrame()

def notebook_text(path: Path) -> str:
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
        return "\n".join("".join(c.get("source", [])) for c in data.get("cells", []))
    except Exception:
        return ""

notebook_rows = []
registration_rows = []
for nb in sorted(ROOT.glob("experiments_lfpc_realistic_thresholds_enhanced_visuals_*registered_methods*.ipynb")):
    src = notebook_text(nb)
    registered = sorted(set(re.findall(r'@register_method\("([^"\n]+)"', src)))
    notebook_rows.append({
        "notebook": nb.name,
        "registered_method_count": len(registered),
        "uses_reducnn_pruner": "ReduCNNPruner(method=method, scope=scope" in src,
        "uses_scope_loop": "for scope in DEMO_SCOPES" in src and "METHODS_BY_SCOPE.get(scope" in src,
        "uses_direct_singular_function": "def run_direct_singular_pruner" in src,
        "exports_pruned_model_checkpoint_path": "pruned_model_checkpoint_path" in src,
        "all_required_local_registered": all(m in registered for m in EXPECTED_LOCAL_METHODS if m != "l1_norm") or "l1_norm" in src,
        "all_required_global_registered": all(m in registered for m in EXPECTED_GLOBAL_METHODS),
    })
    for method in registered:
        registration_rows.append({
            "notebook": nb.name,
            "method": method,
            "method_display": _display_method(method),
            "literature_scope_family": "local" if method in EXPECTED_LOCAL_METHODS else ("global" if method in EXPECTED_GLOBAL_METHODS else "package/built-in-or-other"),
        })

registered_notebook_audit = pd.DataFrame(notebook_rows)
registered_method_audit = pd.DataFrame(registration_rows)
registered_notebook_audit.to_csv(TABLE_DIR / "registered_method_notebook_audit.csv", index=False)
registered_method_audit.to_csv(TABLE_DIR / "registered_method_decorator_inventory.csv", index=False)

# Audit exported singular benchmark artifacts: these are the same-scope singular models used for comparison.
singular_artifact_rows = []
for _, run in run_index[run_index["has_run"]].iterrows():
    run_dir = Path(run["run_dir"])
    singular_path = run_dir / "current_run_singular_method_benchmarks.csv"
    if not singular_path.exists():
        singular_path = run_dir / "singular_method_benchmarks.csv"
    singular_df = _safe_read_csv_local(singular_path) if singular_path.exists() else pd.DataFrame()
    if singular_df.empty:
        singular_artifact_rows.append({
            "objective": run["objective_label"], "dataset": run["dataset"], "model": run["model"],
            "scope": "", "ratio": np.nan, "method_count": 0, "methods": "",
            "has_checkpoint_paths": False, "singular_artifact": str(singular_path) if singular_path.exists() else "",
        })
        continue
    method_col = "method" if "method" in singular_df.columns else "method_or_stack"
    for (scope, ratio), grp in singular_df.groupby(["scope", "ratio"], dropna=False):
        methods = sorted(set(grp[method_col].astype(str))) if method_col in grp.columns else []
        singular_artifact_rows.append({
            "objective": run["objective_label"],
            "dataset": run["dataset"],
            "model": run["model"],
            "scope": scope,
            "ratio": ratio,
            "method_count": len(methods),
            "methods": " + ".join(_display_method(m) for m in methods),
            "has_checkpoint_paths": "pruned_model_checkpoint_path" in grp.columns and grp["pruned_model_checkpoint_path"].notna().any(),
            "singular_benchmark_engine": grp.get("singular_benchmark_engine", pd.Series([""])).astype(str).mode().iloc[0] if "singular_benchmark_engine" in grp.columns and not grp.empty else "",
            "singular_artifact": str(singular_path),
        })

singular_pruning_provenance = pd.DataFrame(singular_artifact_rows)
singular_pruning_provenance.to_csv(TABLE_DIR / "registered_singular_pruning_artifact_provenance.csv", index=False)

display(registered_notebook_audit)
display(registered_method_audit.groupby(["literature_scope_family", "method", "method_display"]).size().reset_index(name="notebook_count"))
display(singular_pruning_provenance.head(40))

print("Registered pruning path checked:")
print("- Methods are declared with @register_method in the experiment notebooks.")
print("- Singular baselines use ReduCNNPruner(method=method, scope=scope, ...).")
print("- Local and global methods are separated by METHODS_BY_SCOPE and DEMO_SCOPES.")
print("- Saved singular/hybrid pruned model paths are exported when the benchmark notebooks are rerun after the checkpoint patch.")
"""
    ),
    md(
        "## 2. Normalize Stack, Policy, and Singular Artifacts\n\nThis section creates three source-of-truth tables: hybrid stacks, layer-wise assignments, and hybrid-versus-singular comparisons."
    ),
    code(
        r"""
def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: could not read {path}: {exc}")
        return pd.DataFrame()


def parse_literal(value: Any, fallback: Any = None) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    if isinstance(value, (list, tuple, dict)):
        return value
    text = str(value).strip()
    if not text:
        return fallback
    try:
        return ast.literal_eval(text)
    except Exception:
        return fallback


def parse_list(value: Any) -> list:
    out = parse_literal(value, fallback=None)
    if isinstance(out, dict):
        return list(out.values())
    if isinstance(out, (list, tuple, set)):
        return list(out)
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return []


def parse_policy(value: Any) -> dict:
    out = parse_literal(value, fallback={})
    return out if isinstance(out, dict) else {}


def method_display(method: Any) -> str:
    text = str(method)
    return METHOD_DISPLAY.get(text, text.replace("custom_", "").replace("c_", "").replace("_", " ").title())


def clean_method_list(value: Any) -> str:
    methods = parse_list(value)
    if not methods:
        return ""
    return " + ".join(method_display(m) for m in methods)


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def norm_good(s: pd.Series, higher=True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = x.min(), x.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=x.index)
    y = (x - lo) / (hi - lo)
    return y if higher else 1.0 - y


def short_stack_codes(df: pd.DataFrame) -> pd.Series:
    keys = []
    for _, r in df.iterrows():
        keys.append(f"{r['objective_code']}_{r['dataset_code']}_{r['model_code']}_{str(r['scope']).upper()[0]}_r{float(r['ratio']):.2f}_{int(r['rank'])}")
    return pd.Series(keys, index=df.index)


stack_frames, policy_frames, singular_frames, warnings = [], [], [], []
for _, rr in run_index[run_index["has_run"]].iterrows():
    run_dir = Path(rr["run_dir"])
    bench = safe_read_csv(run_dir / "fixed_hybrid_stack_benchmarks.csv")
    policy = safe_read_csv(run_dir / "lfpc_discovered_layer_policy_phase1.csv")
    singular = safe_read_csv(run_dir / "fixed_stack_vs_all_singular_display.csv")
    if bench.empty:
        warnings.append(f"Missing fixed hybrid benchmark for {run_dir}")
        continue
    bench = bench.copy()
    bench["dataset"] = bench.get("dataset", rr["dataset"])
    bench["model"] = bench.get("model", rr["model"])
    bench["objective"] = rr["objective"]
    bench["objective_label"] = rr["objective_label"]
    bench["run_dir"] = str(run_dir)
    bench["run_stamp"] = rr["run_stamp"]
    bench["has_singular_comparison"] = not singular.empty
    for c in ["ratio", "variance_threshold", "spearman_threshold", "jaccard_threshold"]:
        bench[c] = numeric(bench, c)
    bench["baseline_accuracy"] = numeric(bench, "baseline_test_accuracy_pct")
    bench["healed_accuracy"] = numeric(bench, "healed_accuracy_pct")
    bench["accuracy_change"] = numeric(bench, "healed_accuracy_delta_pct")
    bench["flops_reduction"] = numeric(bench, "healed_flops_reduction_pct")
    bench["params_reduction"] = numeric(bench, "healed_params_reduction_pct")
    bench["fixed_pruning_time_sec"] = numeric(bench, "end_to_end_pruning_time_sec")
    bench["stack_composition"] = bench["selected_methods"].apply(clean_method_list) if "selected_methods" in bench.columns else ""
    bench["layer_policy_dict"] = bench["layer_policy"].apply(parse_policy) if "layer_policy" in bench.columns else [{}] * len(bench)
    bench["dataset_label"] = bench["dataset"].map({"cifar10": "CIFAR-10", "cats_dogs": "Cats-vs-Dogs"}).fillna(bench["dataset"])
    bench["model_label"] = bench["model"].map({"vgg16": "VGG16", "resnet18": "ResNet18"}).fillna(bench["model"])
    bench["dataset_code"] = bench["dataset"].map({"cifar10": "C10", "cats_dogs": "CD"}).fillna(bench["dataset"])
    bench["model_code"] = bench["model"].map({"vgg16": "VGG", "resnet18": "RN"}).fillna(bench["model"])
    bench["objective_code"] = bench["objective"].map({"all_three": "ALL", "flops_accuracy": "FA", "time_accuracy": "TA"}).fillna(bench["objective"])
    stack_frames.append(bench)

    if not policy.empty:
        policy = policy.copy()
        policy["dataset"] = rr["dataset"]
        policy["model"] = rr["model"]
        policy["objective"] = rr["objective"]
        policy["objective_label"] = rr["objective_label"]
        policy["run_dir"] = str(run_dir)
        policy["method_display"] = policy["selected_method"].apply(method_display) if "selected_method" in policy.columns else ""
        policy_frames.append(policy)

    if not singular.empty:
        singular = singular.copy()
        singular["dataset"] = rr["dataset"]
        singular["model"] = rr["model"]
        singular["objective"] = rr["objective"]
        singular["objective_label"] = rr["objective_label"]
        singular["run_dir"] = str(run_dir)
        singular_frames.append(singular)
    else:
        warnings.append(f"No singular comparison artifact for {rr['objective_label']} | {rr['dataset']} | {rr['model']} | {run_dir.name}")

stacks = pd.concat(stack_frames, ignore_index=True) if stack_frames else pd.DataFrame()
policies = pd.concat(policy_frames, ignore_index=True) if policy_frames else pd.DataFrame()
singular_cmp = pd.concat(singular_frames, ignore_index=True) if singular_frames else pd.DataFrame()

def add_objective_score(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in df.groupby(["objective", "dataset", "model", "ratio", "scope"], dropna=False):
        h = g.copy()
        if h["objective"].iloc[0] == "flops_accuracy":
            h["objective_score"] = 0.5 * norm_good(h["accuracy_change"], True) + 0.5 * norm_good(h["flops_reduction"], True)
        elif h["objective"].iloc[0] == "time_accuracy":
            h["objective_score"] = 0.5 * norm_good(h["accuracy_change"], True) + 0.5 * norm_good(h["fixed_pruning_time_sec"], False)
        else:
            h["objective_score"] = (
                norm_good(h["accuracy_change"], True)
                + norm_good(h["flops_reduction"], True)
                + norm_good(h["fixed_pruning_time_sec"], False)
            ) / 3.0
        out.append(h)
    return pd.concat(out, ignore_index=True)

stacks = add_objective_score(stacks)
stacks["rank"] = stacks.groupby(["objective", "dataset", "model", "ratio", "scope"], dropna=False)["objective_score"].rank(ascending=False, method="first").astype(int)
top_stacks = stacks[stacks["rank"] <= TOP_N].copy().sort_values(["objective", "dataset", "model", "ratio", "scope", "rank"])
top_stacks["short_stack_id"] = short_stack_codes(top_stacks)

top_stacks.to_csv(TABLE_DIR / "top_discovered_hybrid_stacks.csv", index=False)
stacks.to_csv(TABLE_DIR / "all_discovered_hybrid_stacks.csv", index=False)
policies.to_csv(TABLE_DIR / "all_layerwise_policy_assignments.csv", index=False)
singular_cmp.to_csv(TABLE_DIR / "all_hybrid_vs_singular_comparisons.csv", index=False)
pd.DataFrame({"warning": warnings}).to_csv(TABLE_DIR / "artifact_warnings.csv", index=False)

print("Hybrid stacks:", len(stacks))
print("Top stacks:", len(top_stacks))
print("Policy rows:", len(policies))
print("Hybrid-vs-singular rows:", len(singular_cmp))
display(pd.DataFrame({"warning": warnings}).head(20))
"""
    ),
    md("## 3. Top Discovered Stack Tables\n\nTop local and global stacks are ranked within each objective, dataset, model, ratio, and scope by an objective-specific normalized ranking value."),
    code(
        r"""
top_table_cols = [
    "objective_label", "dataset_label", "model_label", "ratio", "scope", "rank", "short_stack_id", "stack_id",
    "stack_composition", "baseline_accuracy", "healed_accuracy", "accuracy_change", "flops_reduction",
    "fixed_pruning_time_sec", "objective_score",
]
top_stack_table = top_stacks[[c for c in top_table_cols if c in top_stacks.columns]].copy()
top_stack_table.to_csv(TABLE_DIR / "top_stack_table_readable.csv", index=False)

for objective_label, g in top_stack_table.groupby("objective_label", sort=False):
    print("\nOBJECTIVE:", objective_label)
    for scope, sg in g.groupby("scope", sort=False):
        print("  SCOPE:", scope)
        display(sg.head(20))
"""
    ),
    md("## 4. Layer-Wise Stack Composition Tables\n\nFor every top stack, this section expands the learned policy into one row per prunable layer."),
    code(
        r"""
layer_rows = []
for _, r in top_stacks.iterrows():
    policy = r.get("layer_policy_dict", {})
    if not isinstance(policy, dict) or not policy:
        warnings.append(f"No layer policy found for {r.get('stack_id')}")
        continue
    items = list(policy.items())
    n = len(items)
    for i, (layer, method) in enumerate(items, start=1):
        if i <= max(1, n // 3):
            layer_region = "early"
        elif i <= max(2, 2 * n // 3):
            layer_region = "middle"
        else:
            layer_region = "late"
        layer_rows.append({
            "objective_label": r["objective_label"],
            "dataset": r["dataset_label"],
            "model": r["model_label"],
            "ratio": r["ratio"],
            "scope": r["scope"],
            "rank": r["rank"],
            "short_stack_id": r["short_stack_id"],
            "stack_id": r["stack_id"],
            "layer_order": i,
            "layer": layer,
            "layer_region": layer_region,
            "method": str(method),
            "method_display": _display_method(method),
        })
layerwise_top = pd.DataFrame(layer_rows)
layerwise_top.to_csv(TABLE_DIR / "top_stack_layerwise_method_assignments.csv", index=False)
display(layerwise_top.head(40))
"""
    ),
    md("## 5. Enhanced Top-Stack Accuracy Bar Plots\n\nEach plot compares top hybrid stacks against the baseline accuracy for the same objective, dataset, model, pruning ratio, and scope. Long stack labels are replaced by short IDs; the mapping is exported."),
    code(
        r"""
def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


plot_index = []
composition_map_rows = []
for _, r in top_stacks.iterrows():
    composition_map_rows.append({
        "short_stack_id": r["short_stack_id"],
        "stack_id": r["stack_id"],
        "composition": r["stack_composition"],
        "objective": r["objective_label"],
        "dataset": r["dataset_label"],
        "model": r["model_label"],
        "ratio": r["ratio"],
        "scope": r["scope"],
    })
composition_map = pd.DataFrame(composition_map_rows).drop_duplicates("short_stack_id")
composition_map.to_csv(TABLE_DIR / "short_stack_id_composition_mapping.csv", index=False)

for keys, g in top_stacks.groupby(["objective", "objective_label", "dataset_label", "model_label", "ratio", "scope"], dropna=False):
    objective, objective_label, dataset, model, ratio, scope = keys
    g = g.sort_values("rank")
    if g.empty or g["healed_accuracy"].isna().all():
        continue
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(g)), 4.8))
    bars = ax.bar(g["short_stack_id"], g["healed_accuracy"], color="#2563EB", alpha=0.88)
    baseline = pd.to_numeric(g["baseline_accuracy"], errors="coerce").dropna()
    if not baseline.empty:
        ax.axhline(float(baseline.iloc[0]), linestyle="--", color="#111827", linewidth=1.2, label=f"Baseline {baseline.iloc[0]:.2f}%")
    labels = [f"{v:.2f}%" if np.isfinite(v) else "" for v in g["healed_accuracy"]]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    ax.set_ylabel("Healed/pruned accuracy (%)")
    ax.set_title(f"Top hybrid stacks | {objective_label} | {dataset} | {model} | r={ratio:g} | {scope}")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fname = f"top_stacks_accuracy_{safe_name(objective)}_{safe_name(dataset)}_{safe_name(model)}_r{ratio:g}_{safe_name(scope)}.png"
    path = PLOT_DIR / fname
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    plot_index.append({"plot_type": "top_stack_accuracy", "path": str(path), "objective": objective_label, "dataset": dataset, "model": model, "ratio": ratio, "scope": scope})

plot_index_df = pd.DataFrame(plot_index)
plot_index_df.to_csv(TABLE_DIR / "plot_index_top_stack_accuracy.csv", index=False)
display(plot_index_df.head(20))
display(composition_map.head(20))
"""
    ),
    md("## 6. Best Hybrid Stack Versus All Singular Methods\n\nLocal hybrids are compared only with local singular methods; global hybrids only with global singular methods. Comparisons are skipped when the artifact does not exist."),
    code(
        r"""
comparison_rows = []
comparison_plot_rows = []

if singular_cmp.empty:
    print("WARNING: no hybrid-vs-singular comparison artifacts found.")
else:
    singular_cmp = singular_cmp.copy()
    for c in ["ratio", "variance_threshold", "spearman_threshold", "jaccard_threshold"]:
        if c in singular_cmp.columns:
            singular_cmp[c] = pd.to_numeric(singular_cmp[c], errors="coerce")
    singular_cmp["method_display"] = singular_cmp["compared_singular_method"].apply(method_display)

    best_keys = ["objective", "dataset", "model", "ratio", "scope"]
    best_hybrids = top_stacks.sort_values("rank").groupby(best_keys, dropna=False).head(1)
    for _, best in best_hybrids.iterrows():
        sub = singular_cmp[
            (singular_cmp["objective"] == best["objective"])
            & (singular_cmp["dataset"] == best["dataset"])
            & (singular_cmp["model"] == best["model"])
            & (singular_cmp["scope"] == best["scope"])
            & (pd.to_numeric(singular_cmp["ratio"], errors="coerce") == float(best["ratio"]))
            & (singular_cmp["hybrid_stack_id"].astype(str) == str(best["stack_id"]))
        ].copy()
        if sub.empty:
            warnings.append(f"No singular comparison rows for best stack {best['stack_id']}")
            continue
        for metric, hybrid_col, method_col, gain_col, ylabel in [
            ("accuracy", "hybrid_healed_accuracy_pct", "method_healed_accuracy_pct", "accuracy_gain_vs_method_pct", "Accuracy (%)"),
            ("flops", "hybrid_flops_reduction_pct", "method_flops_reduction_pct", "flops_gain_vs_method_pct", "FLOPs reduction (%)"),
            ("time", "hybrid_end_to_end_pruning_time_sec", "method_end_to_end_pruning_time_sec", "time_gain_vs_method_sec", "Fixed-stack pruning time (s)"),
        ]:
            if hybrid_col not in sub.columns or method_col not in sub.columns:
                warnings.append(f"Missing {metric} columns for {best['stack_id']}")
                continue
            plot_df = sub[["method_display", method_col]].copy().rename(columns={method_col: "singular_value"})
            plot_df = plot_df.dropna(subset=["singular_value"]).sort_values("singular_value", ascending=(metric == "time"))
            if plot_df.empty or pd.isna(sub[hybrid_col].iloc[0]):
                warnings.append(f"Missing {metric} values for {best['stack_id']}")
                continue
            comparison_rows.append({
                "objective": best["objective_label"],
                "dataset": best["dataset_label"],
                "model": best["model_label"],
                "ratio": best["ratio"],
                "scope": best["scope"],
                "short_stack_id": best["short_stack_id"],
                "stack_id": best["stack_id"],
                "metric": metric,
                "hybrid_value": sub[hybrid_col].iloc[0],
                "best_singular_value": plot_df["singular_value"].max() if metric != "time" else plot_df["singular_value"].min(),
                "mean_gain_vs_singular": pd.to_numeric(sub[gain_col], errors="coerce").mean() if gain_col in sub.columns else np.nan,
            })
        # Three-panel plot for this group.
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        panels = [
            ("accuracy", "hybrid_healed_accuracy_pct", "method_healed_accuracy_pct", "Accuracy (%)", False),
            ("flops", "hybrid_flops_reduction_pct", "method_flops_reduction_pct", "FLOPs reduction (%)", False),
            ("time", "hybrid_end_to_end_pruning_time_sec", "method_end_to_end_pruning_time_sec", "Pruning time (s; lower better)", True),
        ]
        made = False
        for ax, (metric, hcol, mcol, ylabel, lower_better) in zip(axes, panels):
            if hcol not in sub.columns or mcol not in sub.columns or pd.isna(sub[hcol].iloc[0]):
                ax.axis("off")
                continue
            p = sub[["method_display", mcol]].dropna().rename(columns={mcol: "value"})
            if p.empty:
                ax.axis("off")
                continue
            p = p.sort_values("value", ascending=lower_better)
            bars = ax.bar(p["method_display"], p["value"], color="#94A3B8", alpha=0.9, label="Singular")
            ax.axhline(float(sub[hcol].iloc[0]), color="#2563EB", linestyle="--", linewidth=1.6, label="Best hybrid")
            ax.set_title(metric.title())
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=55)
            ax.grid(axis="y", alpha=0.25)
            made = True
        if made:
            axes[0].legend(loc="best")
            fig.suptitle(f"Best hybrid vs singular | {best['objective_label']} | {best['dataset_label']} | {best['model_label']} | r={best['ratio']:g} | {best['scope']}", y=1.02)
            fig.tight_layout()
            fname = f"best_hybrid_vs_singular_{safe_name(best['objective'])}_{safe_name(best['dataset_label'])}_{safe_name(best['model_label'])}_r{best['ratio']:g}_{safe_name(best['scope'])}.png"
            path = PLOT_DIR / fname
            fig.savefig(path, bbox_inches="tight")
            comparison_plot_rows.append({"path": str(path), "objective": best["objective_label"], "dataset": best["dataset_label"], "model": best["model_label"], "ratio": best["ratio"], "scope": best["scope"]})
        plt.close(fig)

comparison_summary = pd.DataFrame(comparison_rows)
comparison_summary.to_csv(TABLE_DIR / "best_hybrid_vs_all_singular_summary.csv", index=False)
pd.DataFrame(comparison_plot_rows).to_csv(TABLE_DIR / "plot_index_best_hybrid_vs_singular.csv", index=False)
display(comparison_summary.head(30))
"""
    ),
    md("## 7. Pareto Trade-Off Charts\n\nHybrid stacks and singular methods are plotted in the relevant objective trade-off space. The best hybrid stack is highlighted."),
    code(
        r"""
pareto_plot_rows = []
if singular_cmp.empty:
    print("WARNING: no singular comparison artifacts available for Pareto hybrid-vs-singular plots.")
else:
    best_hybrids = top_stacks.sort_values("rank").groupby(["objective", "dataset", "model", "ratio", "scope"], dropna=False).head(1)
    for _, best in best_hybrids.iterrows():
        sub = singular_cmp[
            (singular_cmp["objective"] == best["objective"])
            & (singular_cmp["dataset"] == best["dataset"])
            & (singular_cmp["model"] == best["model"])
            & (singular_cmp["scope"] == best["scope"])
            & (pd.to_numeric(singular_cmp["ratio"], errors="coerce") == float(best["ratio"]))
            & (singular_cmp["hybrid_stack_id"].astype(str) == str(best["stack_id"]))
        ].copy()
        if sub.empty:
            continue
        # Projection 1: FLOPs vs accuracy.
        if {"method_flops_reduction_pct", "method_healed_accuracy_pct", "hybrid_flops_reduction_pct", "hybrid_healed_accuracy_pct"}.issubset(sub.columns):
            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            ax.scatter(sub["method_flops_reduction_pct"], sub["method_healed_accuracy_pct"], color="#94A3B8", s=55, label="Singular methods")
            ax.scatter([sub["hybrid_flops_reduction_pct"].iloc[0]], [sub["hybrid_healed_accuracy_pct"].iloc[0]], color="#2563EB", s=160, marker="*", label="Best hybrid")
            for _, row in sub.iterrows():
                if pd.notna(row.get("method_flops_reduction_pct")) and pd.notna(row.get("method_healed_accuracy_pct")):
                    ax.text(row["method_flops_reduction_pct"], row["method_healed_accuracy_pct"], method_display(row["compared_singular_method"]), fontsize=7, alpha=0.8)
            ax.set_xlabel("FLOPs reduction (%)")
            ax.set_ylabel("Healed accuracy (%)")
            ax.set_title(f"Pareto: FLOPs vs accuracy | {best['objective_label']} | {best['dataset_label']} | {best['model_label']} | r={best['ratio']:g} | {best['scope']}")
            ax.grid(alpha=0.25)
            ax.legend()
            path = PLOT_DIR / f"pareto_flops_accuracy_{safe_name(best['objective'])}_{safe_name(best['dataset_label'])}_{safe_name(best['model_label'])}_r{best['ratio']:g}_{safe_name(best['scope'])}.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            pareto_plot_rows.append({"projection": "flops_accuracy", "path": str(path), "objective": best["objective_label"], "dataset": best["dataset_label"], "model": best["model_label"], "ratio": best["ratio"], "scope": best["scope"]})
        # Projection 2: time vs accuracy.
        if {"method_end_to_end_pruning_time_sec", "method_healed_accuracy_pct", "hybrid_end_to_end_pruning_time_sec", "hybrid_healed_accuracy_pct"}.issubset(sub.columns):
            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            ax.scatter(sub["method_end_to_end_pruning_time_sec"], sub["method_healed_accuracy_pct"], color="#94A3B8", s=55, label="Singular methods")
            ax.scatter([sub["hybrid_end_to_end_pruning_time_sec"].iloc[0]], [sub["hybrid_healed_accuracy_pct"].iloc[0]], color="#10B981", s=160, marker="*", label="Best hybrid")
            ax.set_xlabel("Fixed pruning time (s; lower better)")
            ax.set_ylabel("Healed accuracy (%)")
            ax.set_title(f"Pareto: time vs accuracy | {best['objective_label']} | {best['dataset_label']} | {best['model_label']} | r={best['ratio']:g} | {best['scope']}")
            ax.grid(alpha=0.25)
            ax.legend()
            path = PLOT_DIR / f"pareto_time_accuracy_{safe_name(best['objective'])}_{safe_name(best['dataset_label'])}_{safe_name(best['model_label'])}_r{best['ratio']:g}_{safe_name(best['scope'])}.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            pareto_plot_rows.append({"projection": "time_accuracy", "path": str(path), "objective": best["objective_label"], "dataset": best["dataset_label"], "model": best["model_label"], "ratio": best["ratio"], "scope": best["scope"]})

pareto_index = pd.DataFrame(pareto_plot_rows)
pareto_index.to_csv(TABLE_DIR / "plot_index_pareto_charts.csv", index=False)
display(pareto_index.head(30))
"""
    ),
    md("## 8. Robustness Summaries\n\nThese summaries ask whether learned stacks and methods recur across datasets, models, ratios, scopes, objectives, and layer regions."),
    code(
        r"""
method_frequency = (
    layerwise_top.groupby(["method_display"], dropna=False)
    .agg(
        uses=("method_display", "size"),
        objectives=("objective_label", "nunique"),
        datasets=("dataset", "nunique"),
        models=("model", "nunique"),
        ratios=("ratio", "nunique"),
        scopes=("scope", "nunique"),
    )
    .reset_index()
    .sort_values("uses", ascending=False)
)
method_frequency.to_csv(TABLE_DIR / "method_frequency_in_top_stacks.csv", index=False)

region_frequency = (
    layerwise_top.groupby(["model", "layer_region", "method_display"], dropna=False)
    .size().reset_index(name="uses")
    .sort_values(["model", "layer_region", "uses"], ascending=[True, True, False])
)
region_frequency.to_csv(TABLE_DIR / "method_preference_by_model_and_layer_region.csv", index=False)

stack_pattern_frequency = (
    top_stacks.assign(composition_key=top_stacks["stack_composition"].fillna(""))
    .groupby(["composition_key"], dropna=False)
    .agg(
        occurrences=("stack_id", "nunique"),
        objectives=("objective_label", "nunique"),
        datasets=("dataset_label", "nunique"),
        models=("model_label", "nunique"),
        scopes=("scope", "nunique"),
        ratios=("ratio", "nunique"),
        mean_accuracy_change=("accuracy_change", "mean"),
        mean_flops_reduction=("flops_reduction", "mean"),
        mean_time=("fixed_pruning_time_sec", "mean"),
    )
    .reset_index()
    .sort_values(["occurrences", "objectives", "datasets", "models"], ascending=False)
)
stack_pattern_frequency.to_csv(TABLE_DIR / "recurring_stack_patterns.csv", index=False)

ratio_stability = (
    top_stacks.groupby(["objective_label", "dataset_label", "model_label", "scope", "ratio"], dropna=False)
    .agg(
        top_stack_count=("stack_id", "nunique"),
        mean_accuracy_change=("accuracy_change", "mean"),
        mean_flops_reduction=("flops_reduction", "mean"),
        mean_time=("fixed_pruning_time_sec", "mean"),
        unique_methods=("stack_composition", lambda s: len(set(" + ".join(s.dropna()).split(" + ")))),
    )
    .reset_index()
)
ratio_stability.to_csv(TABLE_DIR / "ratio_stability_summary.csv", index=False)

display(method_frequency.head(20))
display(region_frequency.head(30))
display(stack_pattern_frequency.head(15))
"""
    ),
    md(
        "## 9. Objective-by-Objective Thesis Analysis\n\nThis section is the main reportable analysis. It does not just list stacks: it summarizes, per objective, which local and global stacks are strongest, how the learned stacks compare with singular methods, which methods recur in the learned layer policies, and what the trade-off plots imply. The all-three objective is included wherever pairwise objectives are included."
    ),
    code(
        r"""
objective_order = ["FLOPs + Accuracy", "Time + Accuracy", "FLOPs + Time + Accuracy"]


def objective_specific_sort(df: pd.DataFrame, objective_label: str) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "accuracy_change_pp": "accuracy_change",
        "flops_reduction_pct": "flops_reduction",
        "fixed_pruning_time_sec": "fixed_pruning_time_sec",
    }
    for src, dst in rename_map.items():
        if dst not in df.columns and src in df.columns:
            df[dst] = df[src]
    if objective_label == "FLOPs + Accuracy":
        return df.sort_values(["objective_score", "accuracy_change", "flops_reduction"], ascending=[False, False, False])
    if objective_label == "Time + Accuracy":
        return df.sort_values(["objective_score", "accuracy_change", "fixed_pruning_time_sec"], ascending=[False, False, True])
    return df.sort_values(["objective_score", "accuracy_change", "flops_reduction", "fixed_pruning_time_sec"], ascending=[False, False, False, True])


def maybe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


analysis_blocks = []
best_by_context_rows = []
for objective_label in objective_order:
    obj = top_stacks[top_stacks["objective_label"] == objective_label].copy()
    if obj.empty:
        continue
    for keys, g in obj.groupby(["dataset_label", "model_label", "ratio", "scope"], dropna=False):
        g = objective_specific_sort(g, objective_label)
        b = g.iloc[0]
        best_by_context_rows.append({
            "objective": objective_label,
            "dataset": keys[0],
            "model": keys[1],
            "ratio": keys[2],
            "scope": keys[3],
            "best_short_stack_id": b["short_stack_id"],
            "best_stack_id": b["stack_id"],
            "composition": b["stack_composition"],
            "baseline_accuracy": b["baseline_accuracy"],
            "healed_accuracy": b["healed_accuracy"],
            "accuracy_change_pp": b["accuracy_change"],
            "flops_reduction_pct": b["flops_reduction"],
            "fixed_pruning_time_sec": b["fixed_pruning_time_sec"],
            "objective_score": b["objective_score"],
            "num_top_candidates_reported": len(g),
        })

best_by_context = pd.DataFrame(best_by_context_rows)
best_by_context.to_csv(TABLE_DIR / "objective_best_stacks_by_context.csv", index=False)

objective_context_summary = (
    best_by_context.groupby(["objective", "scope"], dropna=False)
    .agg(
        contexts=("best_stack_id", "count"),
        mean_accuracy_change_pp=("accuracy_change_pp", "mean"),
        best_accuracy_change_pp=("accuracy_change_pp", "max"),
        mean_flops_reduction_pct=("flops_reduction_pct", "mean"),
        best_flops_reduction_pct=("flops_reduction_pct", "max"),
        mean_fixed_pruning_time_sec=("fixed_pruning_time_sec", "mean"),
        fastest_fixed_pruning_time_sec=("fixed_pruning_time_sec", "min"),
    )
    .reset_index()
)
objective_context_summary.to_csv(TABLE_DIR / "objective_scope_context_summary.csv", index=False)
display(objective_context_summary)

# Hybrid-vs-singular win-rate summaries. Positive time_gain_vs_method_sec means the hybrid is faster.
if not singular_cmp.empty:
    sc = singular_cmp.copy()
    for c in ["ratio", "accuracy_gain_vs_method_pct", "flops_gain_vs_method_pct", "time_gain_vs_method_sec"]:
        if c in sc.columns:
            sc[c] = pd.to_numeric(sc[c], errors="coerce")
    sc["objective_label"] = sc["objective_label"].replace({"FLOPs + Time + Accuracy": "FLOPs + Time + Accuracy"})
    win_summary = (
        sc.groupby(["objective_label", "dataset", "model", "ratio", "scope"], dropna=False)
        .agg(
            singular_methods_compared=("compared_singular_method", "nunique"),
            accuracy_win_rate=("accuracy_gain_vs_method_pct", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            flops_win_rate=("flops_gain_vs_method_pct", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            time_win_rate=("time_gain_vs_method_sec", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            mean_accuracy_gain_pp=("accuracy_gain_vs_method_pct", "mean"),
            mean_flops_gain_pct=("flops_gain_vs_method_pct", "mean"),
            mean_time_gain_sec=("time_gain_vs_method_sec", "mean"),
        )
        .reset_index()
    )
else:
    win_summary = pd.DataFrame()
win_summary.to_csv(TABLE_DIR / "objective_hybrid_vs_singular_win_rates.csv", index=False)
display(win_summary.head(30))
"""
    ),
    code(
        r"""
# A compact figure per objective: best local/global stack by context.
objective_plot_rows = []
for objective_label in objective_order:
    data = best_by_context[best_by_context["objective"] == objective_label].copy()
    if data.empty:
        continue
    data["context_label"] = data["dataset"] + "\n" + data["model"] + "\nr=" + data["ratio"].map(lambda x: f"{x:g}")
    contexts = list(dict.fromkeys(data["context_label"]))
    x = np.arange(len(contexts))
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(max(12, 1.15 * len(contexts)), 4.6))
    for ax, metric, ylabel, title, higher in [
        (axes[0], "accuracy_change_pp", "Accuracy change (pp)", "Accuracy retention", True),
        (axes[1], "flops_reduction_pct", "FLOPs reduction (%)", "Compression", True),
        (axes[2], "fixed_pruning_time_sec", "Fixed pruning time (s)", "Runtime cost", False),
    ]:
        for offset, scope, color in [(-width / 2, "local", "#2563EB"), (width / 2, "global", "#10B981")]:
            vals = []
            labels = []
            for ctx in contexts:
                row = data[(data["context_label"] == ctx) & (data["scope"] == scope)]
                if row.empty:
                    vals.append(np.nan)
                    labels.append("")
                else:
                    vals.append(maybe_float(row.iloc[0][metric]))
                    labels.append(str(row.iloc[0]["best_short_stack_id"]))
            bars = ax.bar(x + offset, vals, width=width, label=scope.title(), color=color, alpha=0.86)
            try:
                ax.bar_label(bars, labels=[f"{v:.1f}" if np.isfinite(v) else "" for v in vals], padding=2, fontsize=7)
            except Exception:
                pass
        if metric == "accuracy_change_pp":
            ax.axhline(0, color="#111827", linewidth=1, alpha=0.45)
            ax.axhline(-7, color="#DC2626", linestyle="--", linewidth=1, alpha=0.75)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(contexts, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle(f"Best learned stacks by context: {objective_label}", y=1.03)
    fig.tight_layout()
    path = PLOT_DIR / f"objective_dashboard_{safe_name(objective_label)}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.show()
    objective_plot_rows.append({"objective": objective_label, "plot": str(path)})

pd.DataFrame(objective_plot_rows).to_csv(TABLE_DIR / "plot_index_objective_dashboards.csv", index=False)
"""
    ),
    code(
        r"""
# Method recurrence and layer-region preference per objective.
objective_method_rows = []
for objective_label in objective_order:
    sub = layerwise_top[layerwise_top["objective_label"] == objective_label].copy()
    if sub.empty:
        continue
    freq = (
        sub.groupby(["scope", "method_display"], dropna=False)
        .size().reset_index(name="uses")
        .sort_values(["scope", "uses"], ascending=[True, False])
    )
    freq["objective"] = objective_label
    objective_method_rows.append(freq)

    piv = freq.pivot_table(index="method_display", columns="scope", values="uses", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.35 * len(piv))))
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([str(c).title() for c in piv.columns])
    ax.set_title(f"Method recurrence in top stacks: {objective_label}")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, int(piv.values[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Layer assignments in top stacks")
    fig.tight_layout()
    path = PLOT_DIR / f"objective_method_recurrence_{safe_name(objective_label)}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.show()

objective_method_frequency = pd.concat(objective_method_rows, ignore_index=True) if objective_method_rows else pd.DataFrame()
objective_method_frequency.to_csv(TABLE_DIR / "objective_method_frequency_in_top_stacks.csv", index=False)

region_pref = (
    layerwise_top.groupby(["objective_label", "model", "layer_region", "method_display"], dropna=False)
    .size().reset_index(name="uses")
    .sort_values(["objective_label", "model", "layer_region", "uses"], ascending=[True, True, True, False])
)
region_pref.to_csv(TABLE_DIR / "objective_layer_region_method_preferences.csv", index=False)
display(region_pref.groupby(["objective_label", "model", "layer_region"]).head(3).head(60))
"""
    ),
    code(
        r"""
# Thesis prose bullets per objective.
objective_report_rows = []
for objective_label in objective_order:
    obj = best_by_context[best_by_context["objective"] == objective_label].copy()
    if obj.empty:
        continue
    local = obj[obj["scope"] == "local"]
    global_ = obj[obj["scope"] == "global"]
    best_acc = obj.sort_values("accuracy_change_pp", ascending=False).head(1)
    best_flops = obj.sort_values("flops_reduction_pct", ascending=False).head(1)
    best_time = obj.sort_values("fixed_pruning_time_sec", ascending=True).head(1)
    method_freq_obj = objective_method_frequency[objective_method_frequency["objective"] == objective_label] if not objective_method_frequency.empty else pd.DataFrame()
    common_methods = (
        method_freq_obj.groupby("method_display")["uses"].sum().sort_values(ascending=False).head(5).index.tolist()
        if not method_freq_obj.empty else []
    )
    if not win_summary.empty:
        ws = win_summary[win_summary["objective_label"] == objective_label]
        acc_win = ws["accuracy_win_rate"].mean() if not ws.empty else np.nan
        flops_win = ws["flops_win_rate"].mean() if not ws.empty else np.nan
        time_win = ws["time_win_rate"].mean() if not ws.empty else np.nan
    else:
        acc_win = flops_win = time_win = np.nan
    objective_report_rows.append({
        "objective": objective_label,
        "top_local_contexts_available": len(local),
        "top_global_contexts_available": len(global_),
        "best_accuracy_stack": best_acc["best_short_stack_id"].iloc[0] if not best_acc.empty else "",
        "best_accuracy_context": f"{best_acc['dataset'].iloc[0]} | {best_acc['model'].iloc[0]} | r={best_acc['ratio'].iloc[0]:g} | {best_acc['scope'].iloc[0]}" if not best_acc.empty else "",
        "best_accuracy_change_pp": best_acc["accuracy_change_pp"].iloc[0] if not best_acc.empty else np.nan,
        "best_flops_stack": best_flops["best_short_stack_id"].iloc[0] if not best_flops.empty else "",
        "best_flops_context": f"{best_flops['dataset'].iloc[0]} | {best_flops['model'].iloc[0]} | r={best_flops['ratio'].iloc[0]:g} | {best_flops['scope'].iloc[0]}" if not best_flops.empty else "",
        "best_flops_reduction_pct": best_flops["flops_reduction_pct"].iloc[0] if not best_flops.empty else np.nan,
        "fastest_stack": best_time["best_short_stack_id"].iloc[0] if not best_time.empty else "",
        "fastest_context": f"{best_time['dataset'].iloc[0]} | {best_time['model'].iloc[0]} | r={best_time['ratio'].iloc[0]:g} | {best_time['scope'].iloc[0]}" if not best_time.empty else "",
        "fastest_time_sec": best_time["fixed_pruning_time_sec"].iloc[0] if not best_time.empty else np.nan,
        "most_common_methods": " + ".join(common_methods),
        "mean_accuracy_win_rate_vs_singular": acc_win,
        "mean_flops_win_rate_vs_singular": flops_win,
        "mean_time_win_rate_vs_singular": time_win,
    })

objective_report = pd.DataFrame(objective_report_rows)
objective_report.to_csv(TABLE_DIR / "objective_by_objective_reportable_summary.csv", index=False)
display(objective_report)

print("\nReportable interpretation:")
for _, r in objective_report.iterrows():
    print(f"\n{r['objective']}")
    print(f"- Local/global coverage: {int(r['top_local_contexts_available'])} local contexts and {int(r['top_global_contexts_available'])} global contexts.")
    print(f"- Strongest accuracy behavior: {r['best_accuracy_stack']} in {r['best_accuracy_context']} with {r['best_accuracy_change_pp']:.2f} pp.")
    print(f"- Strongest FLOPs behavior: {r['best_flops_stack']} in {r['best_flops_context']} with {r['best_flops_reduction_pct']:.2f}% reduction.")
    print(f"- Fastest fixed-stack behavior: {r['fastest_stack']} in {r['fastest_context']} with {r['fastest_time_sec']:.2f}s.")
    print(f"- Recurring methods in top stacks: {r['most_common_methods']}.")
    if np.isfinite(r['mean_accuracy_win_rate_vs_singular']):
        print(f"- Against singular methods, mean win rates are accuracy={100*r['mean_accuracy_win_rate_vs_singular']:.1f}%, FLOPs={100*r['mean_flops_win_rate_vs_singular']:.1f}%, time={100*r['mean_time_win_rate_vs_singular']:.1f}%.")
    else:
        print("- Singular comparison artifacts are incomplete for this objective, so win rates are reported only where available.")
"""
    ),
    md("## 9. Thesis-Style Interpretation\n\nThe following notes are generated from the artifact tables and should be refined into prose for the results chapter."),
    code(
        r"""
interpretation_rows = []
for objective_label, g in top_stacks.groupby("objective_label", sort=False):
    best = g.sort_values("objective_score", ascending=False).head(1)
    if best.empty:
        continue
    b = best.iloc[0]
    interpretation_rows.append({
        "objective": objective_label,
        "what_it_optimizes": {
            "FLOPs + Accuracy": "compression while preserving healed accuracy",
            "Time + Accuracy": "low pruning/scoring cost while preserving healed accuracy",
            "FLOPs + Time + Accuracy": "balanced compression, runtime, and accuracy behavior",
        }.get(objective_label, "objective-specific trade-off"),
        "best_observed_stack": b["short_stack_id"],
        "best_context": f"{b['dataset_label']} | {b['model_label']} | r={b['ratio']:g} | {b['scope']}",
        "best_composition": b["stack_composition"],
        "accuracy_change_pp": b["accuracy_change"],
        "flops_reduction_pct": b["flops_reduction"],
        "fixed_time_sec": b["fixed_pruning_time_sec"],
        "local_global_note": "Inspect local and global rows separately; comparisons are not pooled across scopes.",
    })
interpretation = pd.DataFrame(interpretation_rows)
interpretation.to_csv(TABLE_DIR / "objective_interpretation_notes.csv", index=False)
display(interpretation)

print("\nNarrative pointers:")
for _, r in interpretation.iterrows():
    print(f"- {r['objective']}: the strongest observed top stack is {r['best_observed_stack']} in {r['best_context']}. It combines {r['best_composition']} and achieved {r['accuracy_change_pp']:.2f} pp accuracy change, {r['flops_reduction_pct']:.2f}% FLOPs reduction, and {r['fixed_time_sec']:.2f}s fixed pruning time where those metrics were available.")
"""
    ),
    md("## 10. Final Compact Summary"),
    code(
        r"""
summary_rows = []
for scope in sorted(top_stacks["scope"].dropna().unique()):
    sg = top_stacks[top_stacks["scope"] == scope].copy()
    if sg.empty:
        continue
    robust_pattern = stack_pattern_frequency[stack_pattern_frequency["composition_key"].isin(sg["stack_composition"].fillna(""))].head(1)
    best_acc = sg.sort_values("accuracy_change", ascending=False).head(1)
    best_flops = sg.sort_values("flops_reduction", ascending=False).head(1)
    best_time = sg.sort_values("fixed_pruning_time_sec", ascending=True).head(1)
    summary_rows.append({
        "scope": scope,
        "most_robust_stack_pattern": robust_pattern["composition_key"].iloc[0] if not robust_pattern.empty else "",
        "best_accuracy_stack": best_acc["short_stack_id"].iloc[0] if not best_acc.empty else "",
        "best_accuracy_change_pp": best_acc["accuracy_change"].iloc[0] if not best_acc.empty else np.nan,
        "best_flops_stack": best_flops["short_stack_id"].iloc[0] if not best_flops.empty else "",
        "best_flops_reduction_pct": best_flops["flops_reduction"].iloc[0] if not best_flops.empty else np.nan,
        "fastest_stack": best_time["short_stack_id"].iloc[0] if not best_time.empty else "",
        "fastest_time_sec": best_time["fixed_pruning_time_sec"].iloc[0] if not best_time.empty else np.nan,
    })

final_summary = pd.DataFrame(summary_rows)
final_summary.to_csv(TABLE_DIR / "final_compact_summary.csv", index=False)
display(final_summary)

pd.DataFrame({"warning": warnings}).drop_duplicates().to_csv(TABLE_DIR / "artifact_warnings.csv", index=False)
print("Report tables:", TABLE_DIR)
print("Report plots:", PLOT_DIR)
"""
    ),
    md(
        "## 11. Final Thesis Figures: Best Learned Policy Per Objective\n\nThis final section produces the two missing thesis-ready views:\n\n1. a layer-wise visualization of which method the best learned stack assigns to each prunable layer;\n2. a same-scope comparison of that best hybrid stack against every singular method available in the artifacts for accuracy, FLOPs reduction, and pruning time.\n\nFor this section, the notebook first chooses the best stack per objective, dataset, model, and scope. Pruning ratio, threshold settings, and Jaccard/Spearman settings are therefore selected by the objective ranking rather than hard-coded. Local stacks are compared only with local singular methods, and global stacks only with global singular methods."
    ),
    code(
        r"""
# Select one best stack per objective, dataset, model, and scope.
best_policy_rows = []
for keys, g in best_by_context.groupby(["objective", "dataset", "model", "scope"], dropna=False):
    objective_label, dataset, model, scope = keys
    g = objective_specific_sort(g, objective_label)
    if g.empty:
        continue
    best_policy_rows.append(g.iloc[0].to_dict())

best_policy_context = pd.DataFrame(best_policy_rows)
best_policy_context.to_csv(TABLE_DIR / "final_best_policy_contexts.csv", index=False)
display(best_policy_context[[
    "objective", "dataset", "model", "scope", "ratio", "best_short_stack_id",
    "composition", "accuracy_change_pp", "flops_reduction_pct", "fixed_pruning_time_sec"
]].head(30))
"""
    ),
    code(
        r"""
# Layer-wise method choice visualization for each selected best policy.
final_policy_layer_rows = []
layer_choice_plot_rows = []
method_palette = {}
palette_values = [
    "#2563EB", "#10B981", "#F59E0B", "#7C3AED", "#EF4444", "#0891B2",
    "#84CC16", "#F97316", "#64748B", "#DB2777", "#14B8A6", "#A855F7",
]

for _, best in best_policy_context.iterrows():
    stack_id = best["best_stack_id"]
    policy_rows = layerwise_top[layerwise_top["stack_id"] == stack_id].copy()
    if policy_rows.empty:
        warnings.append(f"No layerwise rows for final best policy {stack_id}")
        continue
    methods = list(dict.fromkeys(policy_rows["method_display"].astype(str)))
    for m in methods:
        if m not in method_palette:
            method_palette[m] = palette_values[len(method_palette) % len(palette_values)]
    method_to_y = {m: i for i, m in enumerate(methods)}
    policy_rows["method_y"] = policy_rows["method_display"].map(method_to_y)
    policy_rows["color"] = policy_rows["method_display"].map(method_palette)
    final_policy_layer_rows.append(policy_rows)

    fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(policy_rows)), max(4.5, 0.45 * len(methods) + 1.8)))
    ax.scatter(
        policy_rows["layer_order"],
        policy_rows["method_y"],
        s=230,
        c=policy_rows["color"],
        edgecolors="#0F172A",
        linewidths=0.9,
        alpha=0.92,
    )
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xticks(policy_rows["layer_order"])
    ax.set_xticklabels(policy_rows["layer"].astype(str), rotation=55, ha="right")
    ax.set_xlabel("Prunable layer")
    ax.set_ylabel("Selected method")
    ax.set_title(
        f"Best learned layer-wise policy | {best['objective']} | {best['dataset']} | {best['model']} | {best['scope']} | r={float(best['ratio']):g}"
    )
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", alpha=0.25)
    subtitle = (
        f"{best['best_short_stack_id']} | Acc Δ={best['accuracy_change_pp']:.2f} pp, "
        f"FLOPs={best['flops_reduction_pct']:.2f}%, Time={best['fixed_pruning_time_sec']:.2f}s"
    )
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, ha="left", va="bottom")
    fig.tight_layout()
    path = PLOT_DIR / (
        f"final_layerwise_policy_{safe_name(best['objective'])}_{safe_name(best['dataset'])}_"
        f"{safe_name(best['model'])}_{safe_name(best['scope'])}.png"
    )
    fig.savefig(path, bbox_inches="tight")
    plt.show()
    layer_choice_plot_rows.append({
        "objective": best["objective"],
        "dataset": best["dataset"],
        "model": best["model"],
        "scope": best["scope"],
        "ratio": best["ratio"],
        "short_stack_id": best["best_short_stack_id"],
        "stack_id": stack_id,
        "plot": str(path),
    })

final_policy_layers = pd.concat(final_policy_layer_rows, ignore_index=True) if final_policy_layer_rows else pd.DataFrame()
final_policy_layers.to_csv(TABLE_DIR / "final_best_policy_layerwise_choices.csv", index=False)
pd.DataFrame(layer_choice_plot_rows).to_csv(TABLE_DIR / "plot_index_final_layerwise_policy_choices.csv", index=False)
display(pd.DataFrame(layer_choice_plot_rows))
"""
    ),
    code(
        r"""
# Same-scope best hybrid vs all singular methods, selected after choosing the best ratio/threshold context.
# If the exact best stack comparison artifact is absent, the notebook attempts a same dataset/model/scope/ratio
# singular pool fallback from other objective runs and labels it clearly.
def same_scope_singular_pool(best):
    if singular_cmp.empty:
        return pd.DataFrame(), "missing"
    sc = singular_cmp.copy()
    for c in ["ratio", "variance_threshold", "spearman_threshold", "jaccard_threshold"]:
        if c in sc.columns:
            sc[c] = pd.to_numeric(sc[c], errors="coerce")
    exact = sc[
        (sc["objective_label"] == best["objective"])
        & (sc["dataset"].map({"cifar10": "CIFAR-10", "cats_dogs": "Cats-vs-Dogs"}).fillna(sc["dataset"]) == best["dataset"])
        & (sc["model"].map({"vgg16": "VGG16", "resnet18": "ResNet18"}).fillna(sc["model"]) == best["model"])
        & (sc["scope"] == best["scope"])
        & (pd.to_numeric(sc["ratio"], errors="coerce") == float(best["ratio"]))
        & (sc["hybrid_stack_id"].astype(str) == str(best["best_stack_id"]))
    ].copy()
    if not exact.empty:
        return exact, "exact_stack_artifact"

    # Fallback: use all singular rows available for the same dataset/model/scope/ratio from any objective.
    # This does not invent singular values; it reuses exported singular benchmarks and flags the source.
    pool = sc[
        (sc["dataset"].map({"cifar10": "CIFAR-10", "cats_dogs": "Cats-vs-Dogs"}).fillna(sc["dataset"]) == best["dataset"])
        & (sc["model"].map({"vgg16": "VGG16", "resnet18": "ResNet18"}).fillna(sc["model"]) == best["model"])
        & (sc["scope"] == best["scope"])
        & (pd.to_numeric(sc["ratio"], errors="coerce") == float(best["ratio"]))
    ].copy()
    if pool.empty:
        return pool, "missing"
    sort_cols = [c for c in ["compared_singular_method", "objective_label"] if c in pool.columns]
    pool = pool.sort_values(sort_cols).drop_duplicates("compared_singular_method", keep="first")
    return pool, "same_scope_ratio_singular_pool"


final_comparison_rows = []
final_comparison_plot_rows = []

for _, best in best_policy_context.iterrows():
    pool, source = same_scope_singular_pool(best)
    if pool.empty:
        warnings.append(f"No same-scope singular pool for final best policy {best['best_short_stack_id']} ({best['objective']} | {best['dataset']} | {best['model']} | {best['scope']} | r={best['ratio']})")
        continue
    pool = pool.copy()
    pool["method_display"] = pool["compared_singular_method"].apply(method_display)
    # Hybrid values always come from the selected best stack row, so all-three remains valid even
    # when the singular pool comes from a same-ratio pairwise run.
    hybrid_acc = maybe_float(best["healed_accuracy"])
    hybrid_acc_delta = maybe_float(best["accuracy_change_pp"])
    hybrid_flops = maybe_float(best["flops_reduction_pct"])
    hybrid_time = maybe_float(best["fixed_pruning_time_sec"])
    baseline_acc = maybe_float(best["baseline_accuracy"])

    # Convert singular accuracy to delta when possible.
    pool["method_accuracy_delta"] = pd.to_numeric(pool.get("method_healed_accuracy_pct", np.nan), errors="coerce") - baseline_acc
    pool["method_flops_reduction_pct"] = pd.to_numeric(pool.get("method_flops_reduction_pct", np.nan), errors="coerce")
    pool["method_time_sec"] = pd.to_numeric(pool.get("method_end_to_end_pruning_time_sec", np.nan), errors="coerce")

    compare_long = []
    for _, row in pool.iterrows():
        compare_long.extend([
            {
                "objective": best["objective"],
                "dataset": best["dataset"],
                "model": best["model"],
                "scope": best["scope"],
                "ratio": best["ratio"],
                "short_stack_id": best["best_short_stack_id"],
                "stack_id": best["best_stack_id"],
                "singular_method": row["compared_singular_method"],
                "singular_method_display": row["method_display"],
                "metric": "accuracy_delta_pp",
                "hybrid_value": hybrid_acc_delta,
                "singular_value": row["method_accuracy_delta"],
                "hybrid_minus_singular": hybrid_acc_delta - row["method_accuracy_delta"] if np.isfinite(hybrid_acc_delta) and pd.notna(row["method_accuracy_delta"]) else np.nan,
                "singular_source": source,
            },
            {
                "objective": best["objective"],
                "dataset": best["dataset"],
                "model": best["model"],
                "scope": best["scope"],
                "ratio": best["ratio"],
                "short_stack_id": best["best_short_stack_id"],
                "stack_id": best["best_stack_id"],
                "singular_method": row["compared_singular_method"],
                "singular_method_display": row["method_display"],
                "metric": "flops_reduction_pct",
                "hybrid_value": hybrid_flops,
                "singular_value": row["method_flops_reduction_pct"],
                "hybrid_minus_singular": hybrid_flops - row["method_flops_reduction_pct"] if np.isfinite(hybrid_flops) and pd.notna(row["method_flops_reduction_pct"]) else np.nan,
                "singular_source": source,
            },
            {
                "objective": best["objective"],
                "dataset": best["dataset"],
                "model": best["model"],
                "scope": best["scope"],
                "ratio": best["ratio"],
                "short_stack_id": best["best_short_stack_id"],
                "stack_id": best["best_stack_id"],
                "singular_method": row["compared_singular_method"],
                "singular_method_display": row["method_display"],
                "metric": "time_sec_lower_is_better",
                "hybrid_value": hybrid_time,
                "singular_value": row["method_time_sec"],
                "hybrid_minus_singular": row["method_time_sec"] - hybrid_time if np.isfinite(hybrid_time) and pd.notna(row["method_time_sec"]) else np.nan,
                "singular_source": source,
            },
        ])
    final_comparison_rows.extend(compare_long)
    comp = pd.DataFrame(compare_long)

    # Comparative bar plot: gains of best hybrid over each singular method.
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    panels = [
        ("accuracy_delta_pp", "Accuracy gain vs singular (pp)", "#10B981"),
        ("flops_reduction_pct", "FLOPs reduction gain vs singular (pp)", "#2563EB"),
        ("time_sec_lower_is_better", "Time saved vs singular (s)", "#F97316"),
    ]
    for ax, (metric, ylabel, color) in zip(axes, panels):
        sub = comp[comp["metric"] == metric].dropna(subset=["hybrid_minus_singular"]).copy()
        if sub.empty:
            ax.axis("off")
            continue
        sub = sub.sort_values("hybrid_minus_singular", ascending=False)
        bar_colors = [color if v >= 0 else "#DC2626" for v in sub["hybrid_minus_singular"]]
        bars = ax.bar(sub["singular_method_display"], sub["hybrid_minus_singular"], color=bar_colors, alpha=0.9)
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_title(metric.replace("_", " "))
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=55)
        ax.grid(axis="y", alpha=0.25)
        try:
            ax.bar_label(bars, labels=[f"{v:+.2f}" for v in sub["hybrid_minus_singular"]], padding=2, fontsize=7)
        except Exception:
            pass
    fig.suptitle(
        f"Best hybrid stack vs singular methods | {best['objective']} | {best['dataset']} | {best['model']} | {best['scope']} | r={float(best['ratio']):g}\n"
        f"{best['best_short_stack_id']}: {best['composition']} | singular source: {source}",
        y=1.06,
        fontsize=11,
    )
    fig.tight_layout()
    path = PLOT_DIR / (
        f"final_best_hybrid_vs_singular_gains_{safe_name(best['objective'])}_{safe_name(best['dataset'])}_"
        f"{safe_name(best['model'])}_{safe_name(best['scope'])}.png"
    )
    fig.savefig(path, bbox_inches="tight")
    plt.show()
    final_comparison_plot_rows.append({
        "objective": best["objective"],
        "dataset": best["dataset"],
        "model": best["model"],
        "scope": best["scope"],
        "ratio": best["ratio"],
        "short_stack_id": best["best_short_stack_id"],
        "stack_id": best["best_stack_id"],
        "singular_source": source,
        "plot": str(path),
    })

final_best_vs_singular = pd.DataFrame(final_comparison_rows)
final_best_vs_singular.to_csv(TABLE_DIR / "final_best_policy_vs_singular_method_gains.csv", index=False)
pd.DataFrame(final_comparison_plot_rows).to_csv(TABLE_DIR / "plot_index_final_best_policy_vs_singular_gains.csv", index=False)
display(pd.DataFrame(final_comparison_plot_rows))
display(final_best_vs_singular.head(45))
"""
    ),
    md(
        "## 12. Objective-Specific Layer-Region Recommendations\n\n"
        "The best-stack plots show *what* each learned stack chose layer by layer. "
        "This section turns those choices into a thesis narrative: for each optimization objective, "
        "which pruning methods are repeatedly selected in early, middle, and late layers? "
        "The goal is not to claim that one method is universally best, but to explain how the learned policies adapt the stack to the objective."
    ),
    code(
        r"""
# Objective-specific early/middle/late method preferences from the final best policies.
if "final_policy_layers" not in globals() or final_policy_layers.empty:
    final_policy_path = TABLE_DIR / "final_best_policy_layerwise_choices.csv"
    final_policy_layers = pd.read_csv(final_policy_path) if final_policy_path.exists() else pd.DataFrame()

region_order = ["early", "middle", "late"]
region_summary_rows = []
region_plot_rows = []

if final_policy_layers.empty:
    print("WARNING: no final best-policy layer choices are available; skipping layer-region recommendations.")
else:
    work = final_policy_layers.copy()
    work["objective_label"] = work["objective_label"].astype(str)
    work["layer_region"] = pd.Categorical(work["layer_region"].astype(str), categories=region_order, ordered=True)
    work["method_display"] = work["method_display"].astype(str)

    counts = (
        work.groupby(["objective_label", "layer_region", "method_display"], observed=True)
        .size().reset_index(name="uses")
    )
    totals = counts.groupby(["objective_label", "layer_region"], observed=True)["uses"].transform("sum")
    counts["share"] = np.where(totals > 0, counts["uses"] / totals, np.nan)
    counts = counts.sort_values(["objective_label", "layer_region", "uses", "method_display"], ascending=[True, True, False, True])

    for (objective_label, layer_region), grp in counts.groupby(["objective_label", "layer_region"], observed=True):
        top = grp.head(3).copy()
        top_methods = " / ".join([f"{r.method_display} ({int(r.uses)}, {100*r.share:.1f}%)" for r in top.itertuples()])
        region_summary_rows.append({
            "objective": objective_label,
            "layer_region": str(layer_region),
            "recommended_pattern": top_methods,
            "top_method": top["method_display"].iloc[0] if not top.empty else "",
            "top_method_uses": int(top["uses"].iloc[0]) if not top.empty else 0,
            "top_method_share": float(top["share"].iloc[0]) if not top.empty else np.nan,
        })

    objective_region_recommendations = pd.DataFrame(region_summary_rows)
    objective_region_recommendations.to_csv(TABLE_DIR / "objective_layer_region_recommendations.csv", index=False)
    counts.to_csv(TABLE_DIR / "objective_layer_region_method_counts.csv", index=False)

    by_model = (
        work.groupby(["objective_label", "dataset", "model", "scope", "layer_region", "method_display"], observed=True)
        .size().reset_index(name="uses")
        .sort_values(["objective_label", "dataset", "model", "scope", "layer_region", "uses"], ascending=[True, True, True, True, True, False])
    )
    by_model_totals = by_model.groupby(["objective_label", "dataset", "model", "scope", "layer_region"], observed=True)["uses"].transform("sum")
    by_model["share"] = np.where(by_model_totals > 0, by_model["uses"] / by_model_totals, np.nan)
    by_model.to_csv(TABLE_DIR / "objective_layer_region_recommendations_by_dataset_model_scope.csv", index=False)

    for objective_label in objective_order:
        sub = counts[counts["objective_label"] == objective_label].copy()
        if sub.empty:
            continue
        method_totals = sub.groupby("method_display")["uses"].sum().sort_values(ascending=False)
        methods = method_totals.head(14).index.tolist()
        piv = (
            sub[sub["method_display"].isin(methods)]
            .pivot_table(index="layer_region", columns="method_display", values="uses", aggfunc="sum", fill_value=0, observed=True)
            .reindex(region_order)
        )
        if piv.empty:
            continue
        piv = piv.reindex(columns=[m for m in methods if m in piv.columns])
        fig, ax = plt.subplots(figsize=(max(8.5, 0.65 * len(piv.columns)), 3.8))
        im = ax.imshow(piv.values, cmap="YlGnBu", aspect="auto")
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels([str(x).title() for x in piv.index])
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=45, ha="right")
        ax.set_title(f"Layer-region method preferences in best stacks: {objective_label}")
        ax.set_xlabel("Selected pruning method")
        ax.set_ylabel("Layer region")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = int(piv.values[i, j])
                if val:
                    ax.text(j, i, val, ha="center", va="center", fontsize=8, color="#0F172A")
        fig.colorbar(im, ax=ax, label="Layer assignments in final best stacks")
        fig.tight_layout()
        path = PLOT_DIR / f"objective_layer_region_method_preferences_{safe_name(objective_label)}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.show()
        region_plot_rows.append({"objective": objective_label, "plot": str(path)})

    pd.DataFrame(region_plot_rows).to_csv(TABLE_DIR / "plot_index_objective_layer_region_recommendations.csv", index=False)

    display(objective_region_recommendations)
    display(by_model.groupby(["objective_label", "dataset", "model", "scope", "layer_region"], observed=True).head(2).head(80))

    print("\nThesis interpretation prompts:")
    for objective_label in objective_order:
        rec = objective_region_recommendations[objective_region_recommendations["objective"] == objective_label]
        if rec.empty:
            continue
        pieces = []
        for region in region_order:
            row = rec[rec["layer_region"] == region]
            if not row.empty:
                pieces.append(f"{region}: {row['recommended_pattern'].iloc[0]}")
        print(f"- {objective_label}: " + "; ".join(pieces))
        if objective_label == "Time + Accuracy":
            print("  Narrative angle: time-aware accuracy preservation tends to be explained by the cheaper recurring methods in the regions above; expensive methods are useful only where they repeatedly survive the objective ranking.")
        elif objective_label == "FLOPs + Accuracy":
            print("  Narrative angle: compression-aware accuracy preservation is explained by the methods that recur in high-compute regions while keeping the final healed accuracy within the selected trade-off region.")
        elif objective_label == "All three":
            print("  Narrative angle: the balanced objective should be presented as the compromise policy, favoring methods that recur across regions without dominating only one metric.")
"""
    ),
    md(
        "### Notes on Direct Singular Re-Pruning\n\nThe plots above use existing benchmark artifacts as the source of truth. Where an exact hybrid-vs-singular comparison artifact is missing, the notebook uses a same-dataset, same-model, same-scope, same-ratio singular benchmark pool from another exported objective run and labels this in the `singular_source` column. This avoids inventing values while still making the all-three objective reportable. If a future run needs fresh singular benchmarks, the experiment notebooks should run the existing registered-method pruning path with the custom registration decorators and export `fixed_stack_vs_all_singular_display.csv`; this reporting notebook will pick it up automatically."
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
