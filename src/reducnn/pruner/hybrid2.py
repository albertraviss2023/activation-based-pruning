from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MethodEvidence:
    """Cost/compression evidence used to choose among overlapping methods."""

    method: str
    total_time_sec: float = 0.0
    flops_reduction_pct: float = 0.0
    simplicity_rank: Optional[float] = None


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return np.ones_like(arr, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(arr.size, dtype=np.float64)
    return ranks / float(max(arr.size - 1, 1))


def spearman_rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ra = rank_normalize(a)
    rb = rank_normalize(b)
    if ra.size != rb.size or ra.size <= 1:
        return 0.0
    if float(np.std(ra)) < 1e-12 or float(np.std(rb)) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def prune_indices(scores: np.ndarray, ratio: float) -> set[int]:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return set()
    ratio = float(np.clip(ratio, 0.0, 0.95))
    k = max(1, int(round(arr.size * ratio)))
    return set(np.argsort(arr, kind="mergesort")[:k].tolist())


def prune_set_overlap(a: np.ndarray, b: np.ndarray, ratio: float) -> float:
    left = prune_indices(a, ratio)
    right = prune_indices(b, ratio)
    if not left:
        return 0.0
    return float(len(left & right) / max(len(left), 1))


def _component_groups(methods: Sequence[str], edges: Iterable[Tuple[str, str]]) -> List[List[str]]:
    parent = {m: m for m in methods}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)

    groups: Dict[str, List[str]] = {}
    for method in methods:
        groups.setdefault(find(method), []).append(method)
    return [sorted(v) for v in groups.values()]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _evidence_map(evidence: Iterable[Mapping[str, Any] | MethodEvidence]) -> Dict[str, MethodEvidence]:
    out: Dict[str, MethodEvidence] = {}
    for item in evidence:
        if isinstance(item, MethodEvidence):
            out[item.method] = item
            continue
        method = str(item.get("method", "")).strip()
        if not method:
            continue
        time_value = item.get("total_time_sec", None)
        if time_value is None:
            time_value = item.get("method_cost_time_sec", None)
        if time_value is None:
            time_value = item.get("simplicity_time_sec", None)
        if time_value is None:
            time_value = _safe_float(item.get("prune_time_sec")) + _safe_float(item.get("heal_time_sec"))
        flops_value = item.get("flops_reduction_pct", None)
        if flops_value is None:
            flops_value = item.get("healed_flops_reduction_pct", None)
        if flops_value is None:
            flops_value = item.get("median_flops_reduction_pct", None)
        rank_value = item.get("simplicity_rank", None)
        out[method] = MethodEvidence(
            method=method,
            total_time_sec=_safe_float(time_value),
            flops_reduction_pct=_safe_float(flops_value),
            simplicity_rank=None if rank_value is None else _safe_float(rank_value, np.inf),
        )
    return out


def _decision_scores(
    methods: Sequence[str],
    evidence: Mapping[str, MethodEvidence],
    *,
    time_weight: float,
    flops_weight: float,
    simplicity_weight: float,
) -> Dict[str, Dict[str, float]]:
    times = [evidence.get(m, MethodEvidence(m)).total_time_sec for m in methods]
    flops = [evidence.get(m, MethodEvidence(m)).flops_reduction_pct for m in methods]
    ranks = [evidence.get(m, MethodEvidence(m, simplicity_rank=None)).simplicity_rank for m in methods]
    positive_times = [t for t in times if t > 0]
    min_time = min(positive_times) if positive_times else 1.0
    max_flops = max([f for f in flops if f > 0], default=1.0)
    positive_ranks = [r for r in ranks if r is not None and np.isfinite(r) and r > 0]
    max_rank = max(positive_ranks) if positive_ranks else 1.0

    scores: Dict[str, Dict[str, float]] = {}
    for method in methods:
        ev = evidence.get(method, MethodEvidence(method))
        time = ev.total_time_sec
        time_score = min_time / time if time > 0 else 0.0
        flops_score = ev.flops_reduction_pct / max_flops if max_flops > 0 else 0.0
        if ev.simplicity_rank is None or not np.isfinite(ev.simplicity_rank) or ev.simplicity_rank <= 0:
            simplicity_score = 0.0
        else:
            simplicity_score = (max_rank + 1.0 - ev.simplicity_rank) / max_rank
        total = (
            float(time_weight) * time_score
            + float(flops_weight) * flops_score
            + float(simplicity_weight) * simplicity_score
        )
        scores[method] = {
            "decision_score": float(total),
            "time_score": float(time_score),
            "flops_score": float(flops_score),
            "simplicity_score": float(simplicity_score),
            "total_time_sec": float(time),
            "flops_reduction_pct": float(ev.flops_reduction_pct),
            "simplicity_rank": float(ev.simplicity_rank) if ev.simplicity_rank is not None else np.nan,
        }
    return scores


def choose_layerwise_overlap_representatives(
    score_maps: Mapping[str, Mapping[str, np.ndarray]],
    *,
    ratio: float,
    evidence: Iterable[Mapping[str, Any] | MethodEvidence],
    overlap_threshold: float = 0.80,
    correlation_threshold: Optional[float] = None,
    time_weight: float = 0.40,
    flops_weight: float = 0.40,
    simplicity_weight: float = 0.20,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Choose one pruning method per layer from methods that prune the same filters.

    Hybrid-2 is intentionally not a complementary blend. It first finds method
    groups whose prune sets overlap above `overlap_threshold`, then picks one
    representative from an overlapping group. The representative is the method
    with the strongest weighted score over measured total time, FLOPs reduction,
    and prior simplicity rank. If no methods overlap on a layer, the fastest and
    most compressive method is selected, but the row is flagged as `no_overlap`.
    """
    evidence_by_method = _evidence_map(evidence)
    layers = sorted({layer for per_method in score_maps.values() for layer in per_method})
    selected_scores: Dict[str, np.ndarray] = {}
    decision_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    for layer in layers:
        vectors = {
            str(method): np.asarray(per_layer[layer], dtype=np.float64).reshape(-1)
            for method, per_layer in score_maps.items()
            if layer in per_layer and np.asarray(per_layer[layer]).size > 0
        }
        if not vectors:
            continue
        target_size = next(iter(vectors.values())).size
        vectors = {m: v for m, v in vectors.items() if v.size == target_size}
        methods = sorted(vectors)
        scores = _decision_scores(
            methods,
            evidence_by_method,
            time_weight=time_weight,
            flops_weight=flops_weight,
            simplicity_weight=simplicity_weight,
        )

        passing_edges: List[Tuple[str, str]] = []
        for i, a in enumerate(methods):
            for b in methods[i + 1 :]:
                overlap = prune_set_overlap(vectors[a], vectors[b], ratio)
                rho = spearman_rank_correlation(vectors[a], vectors[b])
                passes_overlap = overlap >= float(overlap_threshold)
                passes_correlation = True if correlation_threshold is None else abs(rho) >= float(correlation_threshold)
                passes = bool(passes_overlap and passes_correlation)
                if passes:
                    passing_edges.append((a, b))
                pair_rows.append(
                    {
                        "layer": layer,
                        "method_a": a,
                        "method_b": b,
                        "prune_set_overlap": float(overlap),
                        "spearman_rank_corr": float(rho),
                        "abs_spearman_rank_corr": float(abs(rho)),
                        "passes_overlap_threshold": bool(passes_overlap),
                        "passes_correlation_threshold": bool(passes_correlation),
                        "passes_agreement": bool(passes),
                    }
                )

        groups = _component_groups(methods, passing_edges)
        eligible_groups = [g for g in groups if len(g) >= 2]
        if eligible_groups:
            group_representatives = []
            for group in eligible_groups:
                representative = max(
                    group,
                    key=lambda m: (
                        scores[m]["decision_score"],
                        scores[m]["time_score"],
                        scores[m]["flops_score"],
                        scores[m]["simplicity_score"],
                        m,
                    ),
                )
                group_representatives.append((group, representative))
            chosen_group, chosen = max(
                group_representatives,
                key=lambda item: (
                    scores[item[1]]["decision_score"],
                    len(item[0]),
                    scores[item[1]]["time_score"],
                    scores[item[1]]["flops_score"],
                    item[1],
                ),
            )
            mode = "overlap_representative"
        else:
            chosen_group = [max(
                methods,
                key=lambda m: (
                    scores[m]["decision_score"],
                    scores[m]["time_score"],
                    scores[m]["flops_score"],
                    scores[m]["simplicity_score"],
                    m,
                ),
            )]
            chosen = chosen_group[0]
            mode = "no_overlap_best_evidence"

        selected_scores[layer] = vectors[chosen]
        chosen_mask = prune_indices(vectors[chosen], ratio)
        layer_pair_count = len(methods) * (len(methods) - 1) // 2
        decision_rows.append(
            {
                "layer": layer,
                "mode": mode,
                "chosen_method": chosen,
                "agreement_group": list(chosen_group),
                "agreement_group_size": len(chosen_group),
                "passing_pair_count": len(passing_edges),
                "candidate_pair_count": layer_pair_count,
                "passing_pair_rate": float(len(passing_edges) / layer_pair_count) if layer_pair_count else 0.0,
                "overlap_threshold": float(overlap_threshold),
                "correlation_threshold": correlation_threshold,
                "time_weight": float(time_weight),
                "flops_weight": float(flops_weight),
                "simplicity_weight": float(simplicity_weight),
                "chosen_decision_score": scores[chosen]["decision_score"],
                "chosen_time_score": scores[chosen]["time_score"],
                "chosen_flops_score": scores[chosen]["flops_score"],
                "chosen_simplicity_score": scores[chosen]["simplicity_score"],
                "chosen_total_time_sec": scores[chosen]["total_time_sec"],
                "chosen_flops_reduction_pct": scores[chosen]["flops_reduction_pct"],
                "chosen_simplicity_rank": scores[chosen]["simplicity_rank"],
                "num_channels": target_size,
                "pruned_channels": len(chosen_mask),
                "layer_pruned_ratio": float(len(chosen_mask) / max(target_size, 1)),
                "decision_rule": (
                    "choose one representative from the overlap-connected group; "
                    "rank by 0.4*time score + 0.4*FLOPs score + 0.2*simplicity rank score; "
                    "tie-break toward lower time"
                ),
            }
        )

    return selected_scores, decision_rows, pair_rows
