from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "analysis_algorithm1_similarity_settings_effects.ipynb"


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
# Algorithm 1 Similarity-Setting Effects on LFPC Stack Quality

This notebook consolidates the latest LFPC objective runs for VGG16 and ResNet18 on CIFAR-10 and Cats-vs-Dogs. It answers two thesis questions:

1. **Do Algorithm 1 similarity settings affect the quality of discovered stacks?**
2. **Are the effects consistent across architectures and datasets?**

The analysis uses exported run artifacts rather than re-running pruning. For each run it joins fixed-stack benchmark results with the discovered layer-policy table, then models and visualizes how pruning ratio, Spearman threshold, Jaccard threshold, variance threshold, objective mode, architecture, and dataset relate to accuracy retention, FLOPs reduction, and fixed-stack pruning time.

Terminology note: the experiment notebooks label the threshold-grid similarity stage as Algorithm 2 in some artifacts. In this analysis, the threshold settings are treated as the **Algorithm 1 similarity settings** because they determine method agreement, candidate filtering, and the candidate space passed into LFPC stack discovery.
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

try:
    import statsmodels.formula.api as smf
    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False

ROOT = Path.cwd()
OUT_ROOT = ROOT / "outputs" / "lfpc_hybrid"
REPORT_DIR = ROOT / "reports" / "similarity_settings_analysis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["cifar10", "cats_dogs"]
MODELS = ["vgg16", "resnet18"]
OBJECTIVE_DIRS = {
    "flops_accuracy": "FLOPs + Accuracy",
    "time_accuracy": "Time + Accuracy",
    "time_flops": "Time + FLOPs",
}
BASE_OBJECTIVE = "all_objectives"
BASE_OBJECTIVE_LABEL = "Accuracy + FLOPs + Time"
MAX_ALLOWED_ACCURACY_DROP = 7.0

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})
"""
    ),
    md(
        r"""
## 1. Locate Latest Run Artifacts

For each dataset, model, and objective mode, this cell selects the latest timestamped run directory that contains `fixed_hybrid_stack_benchmarks.csv`. Direct timestamp folders under `outputs/lfpc_hybrid/{dataset}/{model}/` are treated as the balanced/all-objective runs from the `_registered_methods` notebooks.
"""
    ),
    code(
        r"""
def is_timestamp_dir(path: Path) -> bool:
    return bool(re.fullmatch(r"\d{8}_\d{6}", path.name))


def latest_run_for(dataset: str, model: str, objective: str) -> Path | None:
    base = OUT_ROOT / dataset / model
    if objective == BASE_OBJECTIVE:
        candidates = [
            p for p in base.iterdir()
            if p.is_dir() and is_timestamp_dir(p) and (p / "fixed_hybrid_stack_benchmarks.csv").exists()
        ] if base.exists() else []
    else:
        obj_base = base / objective
        candidates = [
            p for p in obj_base.iterdir()
            if p.is_dir() and is_timestamp_dir(p) and (p / "fixed_hybrid_stack_benchmarks.csv").exists()
        ] if obj_base.exists() else []
    return sorted(candidates)[-1] if candidates else None


run_records = []
for dataset in DATASETS:
    for model in MODELS:
        for objective in [BASE_OBJECTIVE, *OBJECTIVE_DIRS.keys()]:
            run_dir = latest_run_for(dataset, model, objective)
            run_records.append({
                "dataset": dataset,
                "model": model,
                "objective": objective,
                "objective_label": BASE_OBJECTIVE_LABEL if objective == BASE_OBJECTIVE else OBJECTIVE_DIRS[objective],
                "run_dir": str(run_dir) if run_dir else "",
                "run_stamp": run_dir.name if run_dir else "",
                "has_run": run_dir is not None,
            })

run_index = pd.DataFrame(run_records)
run_index.to_csv(REPORT_DIR / "run_index.csv", index=False)
display(run_index)
print("Run index saved to", REPORT_DIR / "run_index.csv")
"""
    ),
    md(
        r"""
## 2. Build the Stack-Level Analysis Table

The table joins fixed-stack benchmark metrics with LFPC policy diagnostics. `candidate_count_total` is the total number of layer-method candidates remaining after similarity filtering. `unique_selected_methods` is the number of distinct methods actually used in the final frozen stack.
"""
    ),
    code(
        r"""
def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Skipping unreadable CSV {path}: {exc}")
        return pd.DataFrame()


def parse_maybe_list(value: Any) -> list:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)
        if isinstance(parsed, dict):
            return list(parsed.values())
    except Exception:
        pass
    return [part.strip() for part in text.split(",") if part.strip()]


def numeric_col(df: pd.DataFrame, *names: str, default=np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def condition_cols() -> list[str]:
    return ["scope", "ratio", "variance_threshold", "spearman_threshold", "jaccard_threshold"]


def policy_features(policy_df: pd.DataFrame) -> pd.DataFrame:
    if policy_df.empty:
        return pd.DataFrame()
    df = policy_df.copy()
    for col in ["ratio", "variance_threshold", "spearman_threshold", "jaccard_threshold"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    candidate_col = None
    for name in ["candidate_methods_after_algorithm2", "candidate_methods_after_algorithm1", "candidate_methods"]:
        if name in df.columns:
            candidate_col = name
            break
    df["_candidate_count"] = df[candidate_col].apply(lambda x: len(parse_maybe_list(x))) if candidate_col else np.nan
    group_cols = ["stack_id"] if "stack_id" in df.columns else condition_cols()
    keep_cols = [c for c in group_cols if c in df.columns]
    agg = df.groupby(keep_cols, dropna=False).agg(
        candidate_count_total=("_candidate_count", "sum"),
        candidate_count_mean=("_candidate_count", "mean"),
        candidate_count_min=("_candidate_count", "min"),
        candidate_count_max=("_candidate_count", "max"),
        unique_selected_methods=("selected_method", lambda s: int(pd.Series(s).dropna().astype(str).nunique())),
        policy_layers=("layer", "nunique"),
        mean_selected_probability=("selected_probability", "mean"),
        mean_entropy=("entropy", "mean"),
        fallback_layers=("fallback_applied", lambda s: int(pd.Series(s).astype(str).str.lower().isin(["true", "1"]).sum())),
    ).reset_index()
    return agg


def recipe_features(recipe_df: pd.DataFrame) -> pd.DataFrame:
    if recipe_df.empty or "stack_id" not in recipe_df.columns:
        return pd.DataFrame()
    cols = [
        "stack_id", "num_uncertain_layers", "num_forced_layers", "num_fallback_layers",
        "expected_stack_method_cost_ratio", "expected_stack_flops_reduction_proxy",
        "policy_discovery_time_sec", "algorithm2_time_sec", "lfpc_training_time_sec",
    ]
    cols = [c for c in cols if c in recipe_df.columns]
    out = recipe_df[cols].copy()
    for col in out.columns:
        if col != "stack_id":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.drop_duplicates("stack_id")


stack_frames = []
singular_frames = []

for _, rr in run_index[run_index["has_run"]].iterrows():
    run_dir = Path(rr["run_dir"])
    bench = safe_read_csv(run_dir / "fixed_hybrid_stack_benchmarks.csv")
    policy = safe_read_csv(run_dir / "lfpc_discovered_layer_policy_phase1.csv")
    recipes = safe_read_csv(run_dir / "discovered_fixed_stack_recipes_phase1.csv")
    singular = safe_read_csv(run_dir / "fixed_stack_vs_all_singular_display.csv")
    if bench.empty:
        continue

    bench = bench.copy()
    bench["dataset"] = bench.get("dataset", rr["dataset"])
    bench["model"] = bench.get("model", rr["model"])
    bench["objective"] = rr["objective"]
    bench["objective_label"] = bench.get("objective_label", rr["objective_label"])
    bench["run_dir"] = str(run_dir)
    bench["run_stamp"] = rr["run_stamp"]
    for col in ["ratio", "variance_threshold", "spearman_threshold", "jaccard_threshold"]:
        bench[col] = pd.to_numeric(bench[col], errors="coerce")
    bench["accuracy_delta"] = numeric_col(bench, "healed_accuracy_delta_pct", "accuracy_delta_pct")
    if bench["accuracy_delta"].isna().all() and {"healed_accuracy_pct", "baseline_test_accuracy_pct"}.issubset(bench.columns):
        bench["accuracy_delta"] = pd.to_numeric(bench["healed_accuracy_pct"], errors="coerce") - pd.to_numeric(bench["baseline_test_accuracy_pct"], errors="coerce")
    bench["healed_accuracy_pct"] = numeric_col(bench, "healed_accuracy_pct", "final_test_accuracy")
    bench["flops_reduction_pct"] = numeric_col(bench, "healed_flops_reduction_pct")
    bench["fixed_time_sec"] = numeric_col(bench, "end_to_end_pruning_time_sec")

    pf = policy_features(policy)
    if not pf.empty:
        join_cols = ["stack_id"] if "stack_id" in bench.columns and "stack_id" in pf.columns else [c for c in condition_cols() if c in bench.columns and c in pf.columns]
        bench = bench.merge(pf, on=join_cols, how="left")
    rf = recipe_features(recipes)
    if not rf.empty and "stack_id" in bench.columns:
        bench = bench.merge(rf, on="stack_id", how="left")
    stack_frames.append(bench)

    if not singular.empty:
        singular = singular.copy()
        singular["dataset"] = rr["dataset"]
        singular["model"] = rr["model"]
        singular["objective"] = rr["objective"]
        singular["objective_label"] = rr["objective_label"]
        singular["run_dir"] = str(run_dir)
        singular_frames.append(singular)

analysis_df = pd.concat(stack_frames, ignore_index=True) if stack_frames else pd.DataFrame()
singular_df = pd.concat(singular_frames, ignore_index=True) if singular_frames else pd.DataFrame()

if not analysis_df.empty:
    analysis_df["log_variance"] = np.log(pd.to_numeric(analysis_df["variance_threshold"], errors="coerce").clip(lower=1e-12))
    analysis_df["dataset_label"] = analysis_df["dataset"].map({"cifar10": "CIFAR-10", "cats_dogs": "Cats-vs-Dogs"}).fillna(analysis_df["dataset"])
    analysis_df["model_label"] = analysis_df["model"].map({"vgg16": "VGG16", "resnet18": "ResNet18"}).fillna(analysis_df["model"])
    analysis_df["objective_label_short"] = analysis_df["objective"].map({
        "all_objectives": "All three",
        "flops_accuracy": "FLOPs + Acc.",
        "time_accuracy": "Time + Acc.",
        "time_flops": "Time + FLOPs",
    }).fillna(analysis_df["objective_label"])
    analysis_df["scope_label"] = analysis_df["scope"].astype(str).str.title()
    analysis_df["accuracy_constraint_pass"] = analysis_df["accuracy_delta"] >= -MAX_ALLOWED_ACCURACY_DROP
    analysis_df["setting_id"] = (
        analysis_df["dataset_label"].astype(str) + " | " + analysis_df["model_label"].astype(str) +
        " | " + analysis_df["objective_label_short"].astype(str) + " | " + analysis_df["scope_label"].astype(str) +
        " | r=" + analysis_df["ratio"].round(3).astype(str) +
        ", v=" + analysis_df["variance_threshold"].astype(str) +
        ", rho=" + analysis_df["spearman_threshold"].astype(str) +
        ", J=" + analysis_df["jaccard_threshold"].astype(str)
    )

analysis_path = REPORT_DIR / "similarity_settings_stack_analysis.csv"
analysis_df.to_csv(analysis_path, index=False)
singular_df.to_csv(REPORT_DIR / "similarity_settings_all_singular_comparisons.csv", index=False)

print("Stack rows:", len(analysis_df))
print("Singular-comparison rows:", len(singular_df))
print("Saved", analysis_path)
display(analysis_df.head())
"""
    ),
    md(
        r"""
## 3. Coverage and Minimum Evidence Table

This table shows how much evidence is available per dataset, architecture, objective, and scope. Missing objective modes are kept explicit so the interpretation does not pretend that every combination was run.
"""
    ),
    code(
        r"""
if analysis_df.empty:
    raise RuntimeError("No stack benchmark artifacts were found. Check OUT_ROOT and run_index.")

coverage = (
    analysis_df.groupby(["dataset_label", "model_label", "objective_label_short", "scope_label"], dropna=False)
    .agg(
        stacks=("stack_id", "nunique"),
        rows=("stack_id", "size"),
        ratios=("ratio", lambda s: ", ".join(f"{x:g}" for x in sorted(pd.Series(s).dropna().unique()))),
        spearman=("spearman_threshold", lambda s: ", ".join(f"{x:g}" for x in sorted(pd.Series(s).dropna().unique()))),
        jaccard=("jaccard_threshold", lambda s: ", ".join(f"{x:g}" for x in sorted(pd.Series(s).dropna().unique()))),
        variance=("variance_threshold", lambda s: ", ".join(f"{x:g}" for x in sorted(pd.Series(s).dropna().unique()))),
        mean_accuracy_delta=("accuracy_delta", "mean"),
        mean_flops_reduction=("flops_reduction_pct", "mean"),
        mean_fixed_time=("fixed_time_sec", "mean"),
    )
    .reset_index()
)
coverage.to_csv(REPORT_DIR / "coverage_by_run_scope.csv", index=False)
display(coverage)
"""
    ),
    md(
        r"""
## 4. Threshold Response Heatmaps

The heatmaps answer whether threshold settings affect stack quality. They aggregate over variance thresholds and pruning ratios inside each facet to emphasize setting regions rather than individual grid cells.
"""
    ),
    code(
        r"""
def savefig(name: str):
    path = REPORT_DIR / name
    plt.savefig(path, bbox_inches="tight")
    print("Saved", path)


def threshold_heatmap_grid(metric: str, title: str, cmap: str, filename: str):
    df = analysis_df.dropna(subset=[metric, "spearman_threshold", "jaccard_threshold"]).copy()
    if df.empty:
        print("No data for", metric)
        return
    facets = []
    for dataset in ["CIFAR-10", "Cats-vs-Dogs"]:
        for model in ["VGG16", "ResNet18"]:
            for objective in sorted(df["objective_label_short"].dropna().unique()):
                sub = df[(df["dataset_label"] == dataset) & (df["model_label"] == model) & (df["objective_label_short"] == objective)]
                if not sub.empty:
                    facets.append((dataset, model, objective, sub))
    ncols = 4
    nrows = max(1, math.ceil(len(facets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.35 * nrows), squeeze=False)
    vmin = np.nanpercentile(df[metric], 5)
    vmax = np.nanpercentile(df[metric], 95)
    if metric == "accuracy_delta":
        vmax = max(vmax, 0)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (dataset, model, objective, sub) in zip(axes.ravel(), facets):
        ax.axis("on")
        piv = sub.pivot_table(index="jaccard_threshold", columns="spearman_threshold", values=metric, aggfunc="mean").sort_index(ascending=False).sort_index(axis=1)
        im = ax.imshow(piv.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{x:g}" for x in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{x:g}" for x in piv.index])
        ax.set_xlabel("Spearman threshold")
        ax.set_ylabel("Jaccard threshold")
        ax.set_title(f"{dataset} | {model}\n{objective}", fontsize=10)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="black")
    fig.suptitle(title, y=1.01, fontsize=15)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    cbar.set_label(metric)
    savefig(filename)
    plt.show()


threshold_heatmap_grid("accuracy_delta", "Threshold response: healed accuracy delta (percentage points)", "RdYlGn", "fig1_accuracy_threshold_heatmap.png")
threshold_heatmap_grid("flops_reduction_pct", "Threshold response: FLOPs reduction (%)", "YlGnBu", "fig2_flops_threshold_heatmap.png")
"""
    ),
    md(
        r"""
## 5. Runtime / Cost Response

This plot checks whether pruning time responds to the size of the candidate space and to the number of unique methods selected in the final stack.
"""
    ),
    code(
        r"""
runtime_df = analysis_df.dropna(subset=["fixed_time_sec"]).copy()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
if not runtime_df.empty:
    for objective, sub in runtime_df.groupby("objective_label_short"):
        axes[0].scatter(sub["candidate_count_total"], sub["fixed_time_sec"], s=55, alpha=0.75, label=objective)
        axes[1].scatter(sub["unique_selected_methods"], sub["fixed_time_sec"], s=55, alpha=0.75, label=objective)
    axes[0].set_xlabel("Candidate count after similarity filtering")
    axes[0].set_ylabel("Fixed-stack pruning time (s)")
    axes[0].set_title("Runtime vs candidate-space size")
    axes[1].set_xlabel("Unique selected methods in final stack")
    axes[1].set_ylabel("Fixed-stack pruning time (s)")
    axes[1].set_title("Runtime vs stack method diversity")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[1].legend(title="Objective", bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("fig3_runtime_cost_response.png")
plt.show()
"""
    ),
    md(
        r"""
## 6. Trade-Off and Pareto View

The Pareto-style plot summarizes FLOPs reduction, healed accuracy delta, and fixed-stack pruning time in one view.
"""
    ),
    code(
        r"""
pareto = analysis_df.dropna(subset=["flops_reduction_pct", "accuracy_delta", "fixed_time_sec"]).copy()
markers = {"VGG16": "o", "ResNet18": "s"}
colors = {"All three": "#111827", "FLOPs + Acc.": "#2563EB", "Time + Acc.": "#10B981", "Time + FLOPs": "#F97316"}
fig, ax = plt.subplots(figsize=(11, 7))
for (objective, model), sub in pareto.groupby(["objective_label_short", "model_label"]):
    size = 35 + 220 * (sub["fixed_time_sec"] - pareto["fixed_time_sec"].min()) / max(1e-9, pareto["fixed_time_sec"].max() - pareto["fixed_time_sec"].min())
    ax.scatter(sub["flops_reduction_pct"], sub["accuracy_delta"], s=size, alpha=0.68, marker=markers.get(model, "o"), color=colors.get(objective), label=f"{objective} | {model}", edgecolor="white", linewidth=0.6)
ax.axhline(0, color="#111827", linestyle="-", linewidth=1, alpha=0.4)
ax.axhline(-MAX_ALLOWED_ACCURACY_DROP, color="#DC2626", linestyle="--", linewidth=1.2, label=f"-{MAX_ALLOWED_ACCURACY_DROP:g} pp guard")
ax.set_xlabel("FLOPs reduction (%)")
ax.set_ylabel("Healed accuracy delta vs baseline (pp)")
ax.set_title("Pareto view of discovered stacks across similarity settings")
ax.grid(alpha=0.25)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1)
savefig("fig4_pareto_similarity_settings.png")
plt.show()
"""
    ),
    md(
        r"""
## 7. Interpretable Statistical Models

The models are deliberately simple and thesis-readable. They estimate whether similarity thresholds remain informative after accounting for pruning ratio, objective, architecture, and dataset. Robust standard errors are used when `statsmodels` is available.
"""
    ),
    code(
        r"""
model_df = analysis_df.copy().rename(columns={"spearman_threshold": "rho", "jaccard_threshold": "jaccard", "variance_threshold": "variance"})
model_df["log_variance"] = np.log(pd.to_numeric(model_df["variance"], errors="coerce").clip(lower=1e-12))
model_df["objective_cat"] = model_df["objective_label_short"].astype(str)
model_df["arch_cat"] = model_df["model_label"].astype(str)
model_df["data_cat"] = model_df["dataset_label"].astype(str)

formulas = {
    "accuracy": "accuracy_delta ~ ratio + rho + jaccard + log_variance + C(objective_cat) + C(arch_cat) + C(data_cat) + rho:jaccard + ratio:C(objective_cat)",
    "flops": "flops_reduction_pct ~ ratio + C(objective_cat) + C(arch_cat) + ratio:C(arch_cat) + rho:jaccard",
    "time": "fixed_time_sec ~ ratio + candidate_count_total + unique_selected_methods + C(objective_cat) + C(arch_cat)",
}

def fit_model(name: str, formula: str, required_y: str) -> pd.DataFrame:
    cols = [required_y, "ratio", "rho", "jaccard", "log_variance", "objective_cat", "arch_cat", "data_cat", "candidate_count_total", "unique_selected_methods"]
    df = model_df[[c for c in cols if c in model_df.columns]].dropna(subset=[required_y]).copy()
    if len(df) < 8:
        print(f"Not enough rows for {name} model:", len(df))
        return pd.DataFrame()
    if HAVE_STATSMODELS:
        fitted = smf.ols(formula, data=df).fit(cov_type="HC3")
        table = fitted.summary2().tables[1].reset_index().rename(columns={"index": "term"})
        table["model"] = name
        table["n"] = int(fitted.nobs)
        table["r_squared"] = fitted.rsquared
        print(f"\n{name.upper()} MODEL")
        print(f"n={int(fitted.nobs)}, R^2={fitted.rsquared:.3f}")
        display(table[["model", "term", "Coef.", "Std.Err.", "P>|z|", "n", "r_squared"]].head(25))
        return table
    print("statsmodels unavailable; using built-in NumPy OLS fallback for", name)
    y = pd.to_numeric(df[required_y], errors="coerce")
    if name == "accuracy":
        x = df[["ratio", "rho", "jaccard", "log_variance"]].copy()
        x["rho:jaccard"] = df["rho"] * df["jaccard"]
        for cat in ["objective_cat", "arch_cat", "data_cat"]:
            x = pd.concat([x, pd.get_dummies(df[cat], prefix=cat, drop_first=True, dtype=float)], axis=1)
        for objective_value in sorted(df["objective_cat"].dropna().unique())[1:]:
            x[f"ratio:objective_cat_{objective_value}"] = df["ratio"] * (df["objective_cat"] == objective_value).astype(float)
    elif name == "flops":
        x = df[["ratio"]].copy()
        x["rho:jaccard"] = df["rho"] * df["jaccard"]
        x = pd.concat([x, pd.get_dummies(df["objective_cat"], prefix="objective_cat", drop_first=True, dtype=float)], axis=1)
        arch_dummies = pd.get_dummies(df["arch_cat"], prefix="arch_cat", drop_first=True, dtype=float)
        x = pd.concat([x, arch_dummies], axis=1)
        for col in arch_dummies.columns:
            x[f"ratio:{col}"] = df["ratio"] * arch_dummies[col]
    else:
        x = df[["ratio", "candidate_count_total", "unique_selected_methods"]].copy()
        x = pd.concat([x, pd.get_dummies(df["objective_cat"], prefix="objective_cat", drop_first=True, dtype=float)], axis=1)
        x = pd.concat([x, pd.get_dummies(df["arch_cat"], prefix="arch_cat", drop_first=True, dtype=float)], axis=1)
    x = x.apply(pd.to_numeric, errors="coerce")
    design = pd.concat([pd.Series(1.0, index=x.index, name="Intercept"), x], axis=1)
    valid = design.notna().all(axis=1) & y.notna()
    design = design.loc[valid]
    y = y.loc[valid]
    X = design.to_numpy(dtype=float)
    yy = y.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    pred = X @ beta
    resid = yy - pred
    n = len(yy)
    p = X.shape[1]
    dof = max(1, n - p)
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    t_stat = beta / np.where(se == 0, np.nan, se)
    p_approx = [math.erfc(abs(float(t)) / math.sqrt(2)) if np.isfinite(t) else np.nan for t in t_stat]
    ss_tot = float(((yy - yy.mean()) @ (yy - yy.mean())))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    table = pd.DataFrame({
        "term": design.columns,
        "Coef.": beta,
        "Std.Err.": se,
        "z_or_t": t_stat,
        "P>|z|": p_approx,
        "model": name,
        "n": n,
        "r_squared": r2,
        "engine": "numpy_ols_fallback",
    })
    print(f"\n{name.upper()} MODEL")
    print(f"n={n}, R^2={r2:.3f}, engine=numpy_ols_fallback")
    display(table[["model", "term", "Coef.", "Std.Err.", "P>|z|", "n", "r_squared", "engine"]].head(25))
    return table

coef_tables = [
    fit_model("accuracy", formulas["accuracy"], "accuracy_delta"),
    fit_model("flops", formulas["flops"], "flops_reduction_pct"),
    fit_model("time", formulas["time"], "fixed_time_sec"),
]
coef_df = pd.concat([t for t in coef_tables if not t.empty], ignore_index=True) if any(not t.empty for t in coef_tables) else pd.DataFrame()
coef_df.to_csv(REPORT_DIR / "interpretable_model_coefficients.csv", index=False)
print("Saved model coefficients to", REPORT_DIR / "interpretable_model_coefficients.csv")
"""
    ),
    md(
        r"""
## 8. Objective-Specific Comparison Summary

This table compares objective modes using accuracy retention, FLOPs reduction, pruning time, candidate count, and selected-method diversity.
"""
    ),
    code(
        r"""
objective_summary = (
    analysis_df.groupby(["objective_label_short"], dropna=False)
    .agg(
        stacks=("stack_id", "nunique"),
        mean_accuracy_delta=("accuracy_delta", "mean"),
        best_accuracy_delta=("accuracy_delta", "max"),
        mean_flops_reduction=("flops_reduction_pct", "mean"),
        best_flops_reduction=("flops_reduction_pct", "max"),
        mean_fixed_time_sec=("fixed_time_sec", "mean"),
        fastest_fixed_time_sec=("fixed_time_sec", "min"),
        mean_candidate_count=("candidate_count_total", "mean"),
        mean_unique_selected_methods=("unique_selected_methods", "mean"),
        accuracy_guard_pass_rate=("accuracy_constraint_pass", "mean"),
    )
    .reset_index()
    .sort_values("objective_label_short")
)
objective_summary.to_csv(REPORT_DIR / "objective_summary.csv", index=False)
display(objective_summary)
"""
    ),
    md(
        r"""
## 9. Tuning Guide

This table converts the analysis into practical guidance for maximum accuracy retention, maximum FLOPs under a 7 pp accuracy-drop guard, fastest pruning under the same guard, and balanced deployment.
"""
    ),
    code(
        r"""
def normalized_good(series: pd.Series, higher_is_better=True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=s.index)
    out = (s - lo) / (hi - lo)
    return out if higher_is_better else 1 - out


guide_rows = []
for (dataset, model, objective, scope), sub in analysis_df.groupby(["dataset_label", "model_label", "objective_label_short", "scope_label"], dropna=False):
    sub = sub.copy()
    sub["balanced_score"] = (
        normalized_good(sub["accuracy_delta"], True)
        + normalized_good(sub["flops_reduction_pct"], True)
        + normalized_good(sub["fixed_time_sec"], False)
    ) / 3.0
    candidates = {
        "maximum accuracy retention": sub.sort_values(["accuracy_delta", "flops_reduction_pct"], ascending=[False, False]).head(1),
        "maximum FLOPs under 7 pp guard": sub[sub["accuracy_constraint_pass"]].sort_values(["flops_reduction_pct", "accuracy_delta"], ascending=[False, False]).head(1),
        "fastest under 7 pp guard": sub[sub["accuracy_constraint_pass"]].sort_values(["fixed_time_sec", "accuracy_delta"], ascending=[True, False]).head(1),
        "balanced deployment": sub.sort_values(["balanced_score"], ascending=False).head(1),
    }
    for goal, pick in candidates.items():
        if pick.empty:
            continue
        r = pick.iloc[0]
        guide_rows.append({
            "dataset": dataset,
            "model": model,
            "objective": objective,
            "scope": scope,
            "goal": goal,
            "recommended_ratio": r["ratio"],
            "recommended_variance_threshold": r["variance_threshold"],
            "recommended_spearman_threshold": r["spearman_threshold"],
            "recommended_jaccard_threshold": r["jaccard_threshold"],
            "accuracy_delta_pp": r["accuracy_delta"],
            "flops_reduction_pct": r["flops_reduction_pct"],
            "fixed_time_sec": r["fixed_time_sec"],
            "candidate_count_total": r.get("candidate_count_total", np.nan),
            "unique_selected_methods": r.get("unique_selected_methods", np.nan),
            "stack_id": r.get("stack_id", ""),
        })

tuning_guide = pd.DataFrame(guide_rows)
tuning_guide.to_csv(REPORT_DIR / "fig5_tuning_guide_summary.csv", index=False)
display(tuning_guide)

compact = tuning_guide.groupby(["goal"]).agg(
    median_ratio=("recommended_ratio", "median"),
    median_spearman=("recommended_spearman_threshold", "median"),
    median_jaccard=("recommended_jaccard_threshold", "median"),
    median_variance=("recommended_variance_threshold", "median"),
    median_accuracy_delta=("accuracy_delta_pp", "median"),
    median_flops=("flops_reduction_pct", "median"),
    median_time=("fixed_time_sec", "median"),
).reset_index()

fig, ax = plt.subplots(figsize=(12, max(3.5, 0.55 * len(compact))))
ax.axis("off")
tbl = ax.table(
    cellText=np.round(compact.drop(columns=["goal"]).values, 3),
    rowLabels=compact["goal"],
    colLabels=[c.replace("_", " ") for c in compact.columns if c != "goal"],
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.4)
ax.set_title("Tuning guide summary across available runs", pad=16)
savefig("fig5_tuning_guide_summary.png")
plt.show()
"""
    ),
    md(
        r"""
## 10. Thesis-Ready Interpretation Notes

These generated notes are a first pass. Use them together with the heatmaps and coefficients when writing the final thesis section.
"""
    ),
    code(
        r"""
def describe_effects():
    lines = []
    acc_by_obj = objective_summary.sort_values("mean_accuracy_delta", ascending=False)
    flops_by_obj = objective_summary.sort_values("mean_flops_reduction", ascending=False)
    time_by_obj = objective_summary.sort_values("mean_fixed_time_sec", ascending=True)
    if not acc_by_obj.empty:
        lines.append(f"Highest mean accuracy retention was observed for {acc_by_obj.iloc[0]['objective_label_short']} with mean delta {acc_by_obj.iloc[0]['mean_accuracy_delta']:.2f} pp.")
    if not flops_by_obj.empty:
        lines.append(f"Highest mean FLOPs reduction was observed for {flops_by_obj.iloc[0]['objective_label_short']} with mean reduction {flops_by_obj.iloc[0]['mean_flops_reduction']:.2f}%.")
    if not time_by_obj.empty:
        lines.append(f"Lowest mean fixed-stack pruning time was observed for {time_by_obj.iloc[0]['objective_label_short']} with mean time {time_by_obj.iloc[0]['mean_fixed_time_sec']:.2f} s.")
    corr_cols = ["ratio", "spearman_threshold", "jaccard_threshold", "variance_threshold", "candidate_count_total", "unique_selected_methods"]
    corr = analysis_df[["accuracy_delta", "flops_reduction_pct", "fixed_time_sec", *corr_cols]].corr(numeric_only=True)
    for target in ["accuracy_delta", "flops_reduction_pct", "fixed_time_sec"]:
        vals = corr[target].drop(labels=[target], errors="ignore").dropna()
        if not vals.empty:
            strongest = vals.abs().sort_values(ascending=False).index[0]
            lines.append(f"For {target}, the strongest simple numeric association is with {strongest} (r={vals[strongest]:.2f}).")
    return lines

interpretation_notes = pd.DataFrame({"note": describe_effects()})
interpretation_notes.to_csv(REPORT_DIR / "interpretation_notes.csv", index=False)
for note in interpretation_notes["note"]:
    print("-", note)
"""
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
