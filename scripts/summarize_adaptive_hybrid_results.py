"""Create clearer reports and timing-adjusted plots for adaptive hybrid runs.

The adaptive notebook records method scoring, layer decisions, and demo metrics
in separate files. This post-processor joins those artifacts so reports make the
expensive scoring pass visible and show which methods were selected per layer.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value in ("", None):
        return []
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(v) for v in parsed]
    except Exception:
        pass
    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    if "+" in text:
        return [x.strip() for x in text.split("+") if x.strip()]
    return [text]


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in ("", None):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _natural_key(text: str) -> List[Any]:
    parts: List[Any] = []
    buf = ""
    is_digit = False
    for ch in text:
        if ch.isdigit() != is_digit and buf:
            parts.append(int(buf) if is_digit else buf)
            buf = ""
        is_digit = ch.isdigit()
        buf += ch
    if buf:
        parts.append(int(buf) if is_digit else buf)
    return parts


def _find_latest_run(root: Path) -> Path:
    candidates = [p.parent for p in root.rglob("adaptive_hybrid_demo_matrix.csv")]
    if not candidates:
        raise FileNotFoundError(f"No adaptive_hybrid_demo_matrix.csv found under {root}")
    return max(candidates, key=lambda p: (p / "adaptive_hybrid_demo_matrix.csv").stat().st_mtime)


def _score_times(run_dir: Path) -> Dict[str, float]:
    rows = _read_csv(run_dir / "method_score_timing.csv")
    return {
        str(r.get("method", "")).strip(): _as_float(r.get("score_time_sec"))
        for r in rows
        if str(r.get("status", "ok")).lower() in ("", "ok", "success")
    }


def _add_adjusted_timing(run_dir: Path, score_time_by_method: Dict[str, float]) -> List[Dict[str, Any]]:
    demo_rows = _read_csv(run_dir / "adaptive_hybrid_demo_matrix.csv")
    all_scored_time = sum(score_time_by_method.values())
    adjusted_rows: List[Dict[str, Any]] = []
    for r in demo_rows:
        selected = _as_list(r.get("selected_stack_members"))
        selected_score_time = sum(score_time_by_method.get(m, 0.0) for m in selected)
        prune_time = _as_float(r.get("prune_time_sec"))
        heal_time = _as_float(r.get("heal_time_sec"))
        adjusted = dict(r)
        adjusted["selected_score_time_sec"] = round(selected_score_time, 6)
        adjusted["all_candidate_score_time_sec"] = round(all_scored_time, 6)
        adjusted["prune_plus_selected_score_time_sec"] = round(prune_time + selected_score_time, 6)
        adjusted["total_time_with_selected_scoring_sec"] = round(prune_time + heal_time + selected_score_time, 6)
        adjusted["total_time_with_all_candidate_scoring_sec"] = round(prune_time + heal_time + all_scored_time, 6)
        adjusted_rows.append(adjusted)
    _write_csv(run_dir / "adaptive_hybrid_demo_matrix_adjusted_timing.csv", adjusted_rows)
    (run_dir / "adaptive_hybrid_demo_matrix_adjusted_timing.json").write_text(
        json.dumps(adjusted_rows, indent=2), encoding="utf-8"
    )
    return adjusted_rows


def _plot_adjusted_time(run_dir: Path, adjusted_rows: List[Dict[str, Any]], score_time_by_method: Dict[str, float]) -> None:
    if not adjusted_rows:
        return
    report_dir = run_dir / "report_visuals"
    report_dir.mkdir(exist_ok=True)

    best = sorted(
        adjusted_rows,
        key=lambda r: (
            _as_float(r.get("ratio")),
            -_as_float(r.get("healed_accuracy_delta_pct")),
            _as_float(r.get("total_time_with_selected_scoring_sec")),
        ),
    )[0]
    ratio = best.get("ratio")
    corr = best.get("correlation_threshold")
    compare_path = run_dir / f"adaptive_hybrid_vs_methods_global_r{ratio}_c{corr}.csv"
    if not compare_path.exists():
        return
    rows = _read_csv(compare_path)
    labels: List[str] = []
    base_times: List[float] = []
    score_times: List[float] = []
    for r in rows:
        method = str(r.get("method", ""))
        labels.append(method)
        base_times.append(_as_float(r.get("simplicity_time_sec")))
        if method == "adaptive_hybrid":
            score_times.append(_as_float(best.get("selected_score_time_sec")))
        else:
            score_times.append(score_time_by_method.get(method, 0.0))

    y = np.arange(len(labels))
    fig_h = max(4.0, 0.42 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(y, base_times, label="Recorded prune/surgery + heal", color="#93C5FD")
    ax.barh(y, score_times, left=base_times, label="Scoring pass", color="#F97316")
    totals = np.asarray(base_times) + np.asarray(score_times)
    for i, total in enumerate(totals):
        ax.text(total + max(totals) * 0.01, i, f"{total:.1f}s", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds")
    ax.set_title(f"Timing with scoring included (global, ratio={ratio}, corr={corr})")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(report_dir / "timing_with_scoring_included.png", dpi=180)
    plt.close(fig)


def _plot_layer_selection(run_dir: Path) -> None:
    decisions = _read_csv(run_dir / "layer_decisions.csv")
    if not decisions:
        return
    report_dir = run_dir / "report_visuals"
    report_dir.mkdir(exist_ok=True)

    ratios = sorted({_as_float(r.get("ratio")) for r in decisions})
    thresholds = sorted({_as_float(r.get("correlation_threshold")) for r in decisions})
    target_ratio = 0.3 if 0.3 in ratios else ratios[0]
    target_threshold = 0.8 if 0.8 in thresholds else thresholds[-1]
    rows = [
        r
        for r in decisions
        if abs(_as_float(r.get("ratio")) - target_ratio) < 1e-9
        and abs(_as_float(r.get("correlation_threshold")) - target_threshold) < 1e-9
    ]
    rows.sort(key=lambda r: _natural_key(str(r.get("layer", ""))))
    if not rows:
        return
    methods = sorted({m for r in rows for m in _as_list(r.get("stack"))})
    layer_labels = [str(r.get("layer", "")) for r in rows]

    matrix = np.zeros((len(methods), len(rows)))
    labels = [["" for _ in rows] for _ in methods]
    for j, r in enumerate(rows):
        stack = _as_list(r.get("stack"))
        for method in stack:
            if method in methods:
                i = methods.index(method)
                matrix[i, j] = 1
                labels[i][j] = "in"

    fig_w = max(9.0, 0.55 * len(rows) + 2.5)
    fig_h = max(3.8, 0.42 * len(methods) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(matrix, aspect="auto", cmap=ListedColormap(["#F8FAFC", "#BFDBFE"]), vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(layer_labels)))
    ax.set_xticklabels(layer_labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(rows)):
            if matrix[i, j]:
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7, color="#111827")
    ax.set_xticks(np.arange(-0.5, len(layer_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color="#CBD5E1", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(f"Selected methods by layer (ratio={target_ratio}, corr={target_threshold})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Method")
    fig.tight_layout()
    fig.savefig(report_dir / "selected_methods_by_layer_annotated.png", dpi=180)
    plt.close(fig)

    counts = Counter()
    mode_counts = Counter()
    for r in rows:
        counts.update(_as_list(r.get("stack")))
        mode_counts.update([str(r.get("mode", ""))])
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(methods))))
    values = [counts[m] for m in methods]
    y = np.arange(len(methods))
    ax.barh(y, values, color="#2563EB")
    for i, v in enumerate(values):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlabel("Number of layers where method appears")
    ax.set_title(f"Layer-selection frequency (ratio={target_ratio}, corr={target_threshold})")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(report_dir / "selected_method_frequency_by_layer.png", dpi=180)
    plt.close(fig)


def _write_layer_choice_justification(run_dir: Path, score_time_by_method: Dict[str, float]) -> List[Dict[str, Any]]:
    decisions = _read_csv(run_dir / "layer_decisions.csv")
    pairs = _read_csv(run_dir / "layer_method_pair_agreement.csv")
    stats = _read_csv(run_dir / "layer_method_score_stats.csv")
    if not decisions:
        return []

    report_dir = run_dir / "report_visuals"
    report_dir.mkdir(exist_ok=True)

    pair_index: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for p in pairs:
        key = (
            str(p.get("scope", "")),
            str(p.get("ratio", "")),
            str(p.get("correlation_threshold", "")),
            str(p.get("layer", "")),
        )
        pair_index[key].append(p)

    layer_count_by_method = Counter(str(s.get("method", "")) for s in stats)
    layer_score_time = {
        (str(s.get("method", "")), str(s.get("layer", ""))): _as_float(s.get("score_time_sec_amortized_by_layer"))
        for s in stats
    }

    rows: List[Dict[str, Any]] = []
    for d in decisions:
        scope = str(d.get("scope", ""))
        ratio = str(d.get("ratio", ""))
        threshold = str(d.get("correlation_threshold", ""))
        layer = str(d.get("layer", ""))
        selected = _as_list(d.get("stack")) or _as_list(d.get("selected"))
        simple = set(_as_list(d.get("simple_candidates")))
        complex_methods = set(_as_list(d.get("complex_candidates")))
        eff = _as_dict(d.get("method_efficiency"))
        key = (scope, ratio, threshold, layer)
        relevant_pairs = pair_index.get(key, [])

        if not selected:
            continue

        for method in selected:
            agreeing = []
            any_pairs = []
            for p in relevant_pairs:
                a = str(p.get("method_a", ""))
                b = str(p.get("method_b", ""))
                if method not in (a, b):
                    continue
                other = b if method == a else a
                any_pairs.append((other, p))
                if str(p.get("passes_both_thresholds", "")).lower() == "true":
                    agreeing.append((other, p))

            pair_passes = bool(agreeing)
            if agreeing or any_pairs:
                best_other, best_pair = max(
                    agreeing or any_pairs,
                    key=lambda item: (
                        _as_float(item[1].get("prune_set_overlap")),
                        _as_float(item[1].get("abs_spearman_rank_corr")),
                    ),
                )
                corr = _as_float(best_pair.get("abs_spearman_rank_corr"))
                overlap = _as_float(best_pair.get("prune_set_overlap"))
            else:
                best_other, corr, overlap = "", 0.0, 0.0

            m_eff = _as_dict(eff.get(method, {}))
            other_eff = _as_dict(eff.get(best_other, {})) if best_other else {}
            amortized = layer_score_time.get((method, layer), 0.0)
            if amortized <= 0 and method in score_time_by_method:
                amortized = score_time_by_method[method] / max(layer_count_by_method.get(method, 1), 1)

            if method in simple and best_other in complex_methods and pair_passes:
                reason = "selected simpler agreeing proxy over complex method"
            elif pair_passes:
                reason = "selected as part of compatible stack after agreement checks"
            elif best_other:
                reason = "retained; strongest pair shown did not pass both agreement thresholds"
            else:
                reason = "retained because no selected pair passed both agreement thresholds"

            rows.append({
                "scope": scope,
                "ratio": ratio,
                "correlation_threshold": threshold,
                "layer": layer,
                "selected_method": method,
                "agrees_with": best_other,
                "spearman_abs_corr": round(corr, 6),
                "prune_set_overlap_rate": round(overlap, 6),
                "passes_agreement_thresholds": pair_passes,
                "selected_is_simple": method in simple,
                "other_is_complex": best_other in complex_methods,
                "selected_relative_simplicity_cost": m_eff.get("relative_simplicity_cost", ""),
                "other_relative_simplicity_cost": other_eff.get("relative_simplicity_cost", ""),
                "selected_flops_reduction_pct": m_eff.get("median_flops_reduction_pct", ""),
                "other_flops_reduction_pct": other_eff.get("median_flops_reduction_pct", ""),
                "selected_score_time_sec_for_layer": round(amortized, 6),
                "selected_total_score_time_sec": round(score_time_by_method.get(method, 0.0), 6),
                "choice_reason": reason,
            })

    _write_csv(report_dir / "layer_choice_justification.csv", rows)
    (report_dir / "layer_choice_justification.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    md = [
        "# Layer Choice Justification",
        "",
        "This table is the thesis-facing audit trail: for each layer, it records the chosen method, the strongest agreeing method, the agreement rates, simplicity evidence, compression evidence, and amortized scoring time. Accuracy is intentionally not reported per layer because accuracy is measured after pruning/healing the whole model.",
        "",
        "| scope | ratio | corr threshold | layer | chosen | agrees with | rank corr | prune-overlap | chosen score time/layer | reason |",
        "|---|---:|---:|---|---|---|---:|---:|---:|---|",
    ]
    for r in rows[:120]:
        md.append(
            f"| {r['scope']} | {r['ratio']} | {r['correlation_threshold']} | {r['layer']} | "
            f"{r['selected_method']} | {r['agrees_with']} | {_fmt(_as_float(r['spearman_abs_corr']))} | "
            f"{_fmt(_as_float(r['prune_set_overlap_rate']))} | {_fmt(_as_float(r['selected_score_time_sec_for_layer']), 4)}s | "
            f"{r['choice_reason']} |"
        )
    if len(rows) > 120:
        md.append(f"\nShowing first 120 of {len(rows)} rows. See `layer_choice_justification.csv` for all rows.")
    (report_dir / "layer_choice_justification.md").write_text("\n".join(md), encoding="utf-8")
    return rows


def _condition_tag(row: Dict[str, Any]) -> str:
    return f"{row.get('scope', 'global')}_r{row.get('ratio')}_c{row.get('correlation_threshold')}"


def _bool_value(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def _stack_condition_rows(run_dir: Path, adjusted_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in adjusted_rows:
        tag = _condition_tag(row)
        effectiveness_path = run_dir / f"adaptive_hybrid_stack_effectiveness_{tag}.csv"
        effectiveness = _read_csv(effectiveness_path)
        selected = _as_list(row.get("selected_stack_members"))
        methods_beaten_accuracy = [
            r.get("compared_method", "")
            for r in effectiveness
            if _bool_value(r.get("accuracy_retained_or_better"))
        ]
        methods_beaten_flops = [
            r.get("compared_method", "")
            for r in effectiveness
            if _bool_value(r.get("flops_reduction_better_or_equal"))
        ]
        methods_beaten_params = [
            r.get("compared_method", "")
            for r in effectiveness
            if _bool_value(r.get("params_reduction_better_or_equal"))
        ]
        methods_beaten_recorded_time = [
            r.get("compared_method", "")
            for r in effectiveness
            if _bool_value(r.get("simplicity_faster"))
        ]
        compare_count = len(effectiveness)
        out.append(
            {
                "condition": tag,
                "scope": row.get("scope", ""),
                "ratio": row.get("ratio", ""),
                "correlation_threshold": row.get("correlation_threshold", ""),
                "stack": " + ".join(selected),
                "stack_size": len(selected),
                "num_masked_layers": row.get("num_masked_layers", ""),
                "healed_accuracy_pct": row.get("healed_accuracy_pct", ""),
                "healed_accuracy_delta_pct": row.get("healed_accuracy_delta_pct", ""),
                "healed_flops_reduction_pct": row.get("healed_flops_reduction_pct", ""),
                "healed_params_reduction_pct": row.get("healed_params_reduction_pct", ""),
                "recorded_prune_heal_time_sec": row.get("simplicity_time_sec", ""),
                "selected_score_time_sec": row.get("selected_score_time_sec", ""),
                "total_time_with_selected_scoring_sec": row.get("total_time_with_selected_scoring_sec", ""),
                "compared_individual_methods": compare_count,
                "accuracy_win_count": len(methods_beaten_accuracy),
                "flops_win_count": len(methods_beaten_flops),
                "params_win_count": len(methods_beaten_params),
                "recorded_time_win_count": len(methods_beaten_recorded_time),
                "accuracy_win_rate": round(len(methods_beaten_accuracy) / compare_count, 4) if compare_count else "",
                "flops_win_rate": round(len(methods_beaten_flops) / compare_count, 4) if compare_count else "",
                "params_win_rate": round(len(methods_beaten_params) / compare_count, 4) if compare_count else "",
                "recorded_time_win_rate": round(len(methods_beaten_recorded_time) / compare_count, 4) if compare_count else "",
                "methods_beaten_accuracy": " | ".join(methods_beaten_accuracy),
                "methods_beaten_flops": " | ".join(methods_beaten_flops),
                "methods_beaten_params": " | ".join(methods_beaten_params),
                "methods_beaten_recorded_time": " | ".join(methods_beaten_recorded_time),
            }
        )
    return out


def _plot_stack_outperformance(run_dir: Path, stack_rows: List[Dict[str, Any]]) -> None:
    report_dir = run_dir / "report_visuals"
    report_dir.mkdir(exist_ok=True)
    rows = [r for r in stack_rows if _as_float(r.get("ratio")) == 0.3 and _as_float(r.get("compared_individual_methods")) > 0]
    if not rows:
        rows = [r for r in stack_rows if _as_float(r.get("compared_individual_methods")) > 0]
    if not rows:
        return
    labels = [f"r{r['ratio']} c{r['correlation_threshold']}" for r in rows]
    metrics = [
        ("accuracy_win_rate", "Accuracy"),
        ("flops_win_rate", "FLOPs"),
        ("params_win_rate", "Params"),
        ("recorded_time_win_rate", "Prune+heal time"),
    ]
    data = np.asarray([[_as_float(r.get(k)) for k, _ in metrics] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.45 * len(rows) + 1.5)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j] * 100:.0f}%", ha="center", va="center", fontsize=9)
    ax.set_title("Hybrid stack win rate against individual stack members")
    fig.colorbar(im, ax=ax, label="Win rate")
    fig.tight_layout()
    fig.savefig(report_dir / "stack_outperformance_win_rates.png", dpi=180)
    plt.close(fig)


def _stack_report(run_dir: Path, stack_rows: List[Dict[str, Any]], decisions: List[Dict[str, str]]) -> None:
    report_dir = run_dir / "report_visuals"
    report_dir.mkdir(exist_ok=True)
    _write_csv(report_dir / "stack_effectiveness_by_condition.csv", stack_rows)
    (report_dir / "stack_effectiveness_by_condition.json").write_text(
        json.dumps(stack_rows, indent=2), encoding="utf-8"
    )

    best_overall = sorted(
        [r for r in stack_rows if _as_float(r.get("compared_individual_methods")) > 0],
        key=lambda r: (
            -_as_float(r.get("accuracy_win_rate")),
            -_as_float(r.get("flops_win_rate")),
            -_as_float(r.get("params_win_rate")),
            _as_float(r.get("total_time_with_selected_scoring_sec")),
        ),
    )
    best_acc = max(stack_rows, key=lambda r: _as_float(r.get("healed_accuracy_delta_pct"))) if stack_rows else {}

    lines = [
        "# Hybrid Stack Effectiveness Report",
        "",
        "Research question: which method combinations work best together, and do they outperform the individual methods?",
        "",
        "## Short Answer",
        "",
    ]
    scored_scopes = sorted({str(r.get("scope", "")).strip() for r in decisions if str(r.get("scope", "")).strip()})
    scored_methods = sorted({m for r in decisions for m in _as_list(r.get("candidate_methods"))})
    simple_scored = sorted({m for r in decisions for m in _as_list(r.get("simple_candidates"))})
    efficiency_rows = _read_csv(run_dir / "method_efficiency_ranking.csv")
    available_local = [r.get("method", "") for r in efficiency_rows if r.get("scope") == "local"]
    available_global = [r.get("method", "") for r in efficiency_rows if r.get("scope") == "global"]
    if best_acc:
        lines.append(
            f"- Strongest accuracy condition: `{best_acc.get('stack')}` at ratio={best_acc.get('ratio')}, "
            f"corr={best_acc.get('correlation_threshold')} reached healed accuracy "
            f"{_fmt(_as_float(best_acc.get('healed_accuracy_pct')))}% "
            f"({_fmt(_as_float(best_acc.get('healed_accuracy_delta_pct')))} pp vs baseline)."
        )
    if best_overall:
        row = best_overall[0]
        lines.append(
            f"- Best balanced stack condition with individual-method comparisons: `{row.get('stack')}` at "
            f"ratio={row.get('ratio')}, corr={row.get('correlation_threshold')}. "
            f"It matched or beat {row.get('accuracy_win_count')}/{row.get('compared_individual_methods')} "
            f"members on accuracy retention, {row.get('flops_win_count')}/{row.get('compared_individual_methods')} "
            f"on FLOPs reduction, {row.get('params_win_count')}/{row.get('compared_individual_methods')} "
            f"on parameter reduction, and {row.get('recorded_time_win_count')}/{row.get('compared_individual_methods')} "
            "on recorded prune+heal time."
        )
    lines += [
        "- For this run, inspect `scope` before interpreting the stack. Older outputs may contain only a "
        "global-compatible stack, while newer notebook runs include both local-compatible and "
        "global-compatible stack grids.",
        f"- Scored scopes in this run: `{', '.join(scored_scopes) if scored_scopes else 'unknown'}`. "
        "If a method is missing, check whether it was actually scored as a compatible candidate before "
        "claiming it failed the correlation test.",
        "- Timing caveat: the base recorded prune+heal time may not include the candidate scoring pass. "
        "The adjusted timing table adds selected scoring time and all-candidate scoring time.",
        "",
        "## Candidate Coverage Check",
        "",
        "The candidate methods that actually appeared in the layer decisions were:",
        "",
        ", ".join(f"`{m}`" for m in scored_methods) if scored_methods else "`unknown`",
        "",
        "The methods available in local-scope efficiency evidence were:",
        "",
        ", ".join(f"`{m}`" for m in available_local) if available_local else "`none found`",
        "",
        "The methods available in global-scope efficiency evidence were:",
        "",
        ", ".join(f"`{m}`" for m in available_global) if available_global else "`none found`",
        "",
        "A clean experiment grid is:",
        "",
        "- `local`: test every local-compatible method and report the selected stack by layer.",
        "- `global`: test every global-compatible method, including simple methods that support global thresholding, and report the selected stack by layer.",
        "- `simple_only`: when only simple methods are compatible, allow and report their own weighted stack instead of forcing a single winner.",
        "",
        "For older global-only evidence, avoid saying simple methods do not work. Say: `Those methods were not in the tested candidate set, so their stack compatibility remains untested in that run.`",
        "",
        "## Tested Stack Conditions",
        "",
        "| ratio | corr | stack | acc delta | FLOPs red. | params red. | accuracy wins | FLOPs wins | params wins | recorded time wins | adjusted total time |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in stack_rows:
        compared = r.get("compared_individual_methods") or "0"
        lines.append(
            f"| {r.get('ratio')} | {r.get('correlation_threshold')} | {r.get('stack')} | "
            f"{_fmt(_as_float(r.get('healed_accuracy_delta_pct')))} | "
            f"{_fmt(_as_float(r.get('healed_flops_reduction_pct')))} | "
            f"{_fmt(_as_float(r.get('healed_params_reduction_pct')))} | "
            f"{r.get('accuracy_win_count')}/{compared} | {r.get('flops_win_count')}/{compared} | "
            f"{r.get('params_win_count')}/{compared} | {r.get('recorded_time_win_count')}/{compared} | "
            f"{_fmt(_as_float(r.get('total_time_with_selected_scoring_sec')))}s |"
        )

    lines += [
        "",
        "## How To Read This",
        "",
        "- `accuracy wins` means the hybrid retained baseline accuracy at least as well as the individual method.",
        "- `FLOPs wins` and `params wins` mean the hybrid compressed at least as much as the individual method.",
        "- `recorded time wins` uses the notebook's prune/surgery plus healing time. Use the adjusted total-time column when discussing real end-to-end scoring cost.",
        "- Conditions with `0` compared methods were run at ratios where same-ratio individual method rows were not available, so they are weaker evidence for the research question.",
        "",
    ]
    if decisions:
        target = [
            r
            for r in decisions
            if abs(_as_float(r.get("ratio")) - 0.3) < 1e-9
            and abs(_as_float(r.get("correlation_threshold")) - 0.8) < 1e-9
        ]
        if target:
            layers = sorted([str(r.get("layer", "")) for r in target], key=_natural_key)
            lines += [
                "## What The Stack Scored",
                "",
                "For ratio=0.3 and corr=0.8, the four-method stack scored/pruned these layers:",
                "",
                ", ".join(f"`{layer}`" for layer in layers),
                "",
            ]
    lines += [
        "## New Files",
        "",
        "- `report_visuals/stack_effectiveness_by_condition.csv/json`",
        "- `report_visuals/stack_outperformance_win_rates.png`",
        "",
    ]
    (report_dir / "hybrid_stack_effectiveness_report.md").write_text("\n".join(lines), encoding="utf-8")


def _brief_report(run_dir: Path, adjusted_rows: List[Dict[str, Any]], score_time_by_method: Dict[str, float]) -> None:
    demo_rows = adjusted_rows
    decisions = _read_csv(run_dir / "layer_decisions.csv")
    report_dir = run_dir / "report_visuals"
    report_dir.mkdir(exist_ok=True)

    best_acc = max(demo_rows, key=lambda r: _as_float(r.get("healed_accuracy_delta_pct"))) if demo_rows else {}
    fastest = min(demo_rows, key=lambda r: _as_float(r.get("total_time_with_selected_scoring_sec"))) if demo_rows else {}
    by_ratio: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in demo_rows:
        by_ratio[str(r.get("ratio"))].append(r)

    lines = [
        "# Adaptive Hybrid Brief Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Main Findings",
        "",
    ]
    if best_acc:
        stack = " + ".join(_as_list(best_acc.get("selected_stack_members")))
        lines.append(
            "- Best healed accuracy delta: "
            f"+{_fmt(_as_float(best_acc.get('healed_accuracy_delta_pct')))} pp at "
            f"ratio={best_acc.get('ratio')}, corr={best_acc.get('correlation_threshold')} "
            f"using `{stack}`."
        )
    if fastest:
        lines.append(
            "- Fastest adjusted hybrid run: "
            f"{_fmt(_as_float(fastest.get('total_time_with_selected_scoring_sec')))}s including selected scoring, "
            f"at ratio={fastest.get('ratio')}, corr={fastest.get('correlation_threshold')}."
        )
    if score_time_by_method:
        slowest_method, slowest_time = max(score_time_by_method.items(), key=lambda kv: kv[1])
        lines.append(
            f"- Scoring is the timing bottleneck: `{slowest_method}` took {_fmt(slowest_time)}s to score, "
            "so timing plots must include scoring separately from surgery/prune and healing."
        )

    lines += [
        "",
        "## Timing Interpretation",
        "",
        "The original `simplicity_time_sec` in the adaptive demo matrix records the demo pruning/surgery step plus healing. "
        "It does not include the earlier candidate method scoring pass. The adjusted table adds:",
        "",
        "- `selected_score_time_sec`",
        "- `prune_plus_selected_score_time_sec`",
        "- `total_time_with_selected_scoring_sec`",
        "- `total_time_with_all_candidate_scoring_sec`",
        "",
    ]

    if decisions:
        target = [
            r
            for r in decisions
            if abs(_as_float(r.get("ratio")) - 0.3) < 1e-9
            and abs(_as_float(r.get("correlation_threshold")) - 0.8) < 1e-9
        ]
        if target:
            mode_counts = Counter(str(r.get("mode", "")) for r in target)
            method_counts = Counter(m for r in target for m in _as_list(r.get("stack")))
            lines += [
                "## Layer Selection",
                "",
                "At ratio=0.3 and corr=0.8, the layer decisions are dominated by:",
                "",
            ]
            for method, count in method_counts.most_common():
                lines.append(f"- `{method}` in {count} layers")
            lines += ["", "Decision modes:"]
            for mode, count in mode_counts.most_common():
                lines.append(f"- `{mode}`: {count} layers")
            lines.append("")

    lines += [
        "## New Files",
        "",
        "- `adaptive_hybrid_demo_matrix_adjusted_timing.csv/json`",
        "- `report_visuals/timing_with_scoring_included.png`",
        "- `report_visuals/selected_methods_by_layer_annotated.png`",
        "- `report_visuals/selected_method_frequency_by_layer.png`",
        "- `report_visuals/layer_choice_justification.csv/json/md`",
        "- `report_visuals/hybrid_stack_effectiveness_report.md`",
        "- `report_visuals/stack_outperformance_win_rates.png`",
        "",
    ]
    (report_dir / "adaptive_hybrid_brief_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Adaptive hybrid run directory. Defaults to latest under outputs/adaptive_hybrid.",
    )
    parser.add_argument("--root", type=Path, default=Path("outputs/adaptive_hybrid"))
    args = parser.parse_args()
    run_dir = args.run_dir or _find_latest_run(args.root)
    run_dir = run_dir.resolve()
    score_time_by_method = _score_times(run_dir)
    adjusted_rows = _add_adjusted_timing(run_dir, score_time_by_method)
    _plot_adjusted_time(run_dir, adjusted_rows, score_time_by_method)
    _plot_layer_selection(run_dir)
    _write_layer_choice_justification(run_dir, score_time_by_method)
    stack_rows = _stack_condition_rows(run_dir, adjusted_rows)
    decisions = _read_csv(run_dir / "layer_decisions.csv")
    _plot_stack_outperformance(run_dir, stack_rows)
    _stack_report(run_dir, stack_rows, decisions)
    _brief_report(run_dir, adjusted_rows, score_time_by_method)
    print(f"Wrote adjusted report artifacts under {run_dir}")


if __name__ == "__main__":
    main()
