import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from time import perf_counter
from ..core.adapter import FrameworkAdapter

class HybridMetaPruner:
    """Implements literature-grounded hybrid pruning metrics.
    
    This engine adaptively blends multiple pruning metrics (Structural, Data-Driven, 
    and Sensitivity-based) based on the relative depth of the layer.
    """
    
    def __init__(self, adapter: FrameworkAdapter, mode: str = 'smooth'):
        """
        Args:
            adapter: Framework-specific adapter.
            mode: 'bucket' (hard thresholds) or 'smooth' (linear interpolation).
        """
        self.adapter = adapter
        self.mode = mode.lower().strip()
        self.config = getattr(adapter, "config", {}) or {}
        self.timing_report: Dict[str, float] = {}
        self.last_metric_weights: Dict[str, Dict[str, float]] = {}
        self.last_metric_contributions: Dict[str, Dict[str, float]] = {}
        self._measured_method_costs: Optional[Dict[str, float]] = None
        triplet = self.config.get("hybrid_metric_triplet", ["l1_norm", "mean_abs_act", "apoz"])
        if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
            triplet = ["l1_norm", "mean_abs_act", "apoz"]
        self.metric_triplet = [str(m).lower().strip() for m in triplet]
        self.last_layer_decisions: Dict[str, Dict[str, Any]] = {}

    def _timing_gate_cfg(self) -> Tuple[str, float, bool]:
        """Returns timing gate mode, max ratio, and whether baseline timing is enabled."""
        mode = str(self.config.get("hybrid_timing_gate", "warn")).lower().strip()
        if mode in ("none", "off", "false", "0", ""):
            mode = "off"
        elif mode in ("true", "1", "yes"):
            mode = "warn"
        elif mode not in ("warn", "error"):
            mode = "warn"
        max_ratio = float(self.config.get("hybrid_timing_max_ratio", 2.0))
        measure_baseline = bool(self.config.get("hybrid_measure_taylor_baseline", mode != "off"))
        return mode, max_ratio, measure_baseline

    def calculate_hybrid_scores(self, model: Any, loader: Any) -> Dict[str, np.ndarray]:
        """Calculates blended consensus scores for every filter."""
        # 1. Get graph and determine prunable layers in order
        graph = self.adapter.trace_graph(model)
        nodes = graph["nodes"]
        # Filter only conv layers and sort by topological order (if possible)
        # For simplicity, we'll use the order they appear in the trace
        prunable_layers = [n for n, d in nodes.items() if d.get("type") == "conv2d"]
        num_layers = len(prunable_layers)
        
        # 2. Get all required scores in one pass (if optimized)
        m_early, m_mid, m_late = self.metric_triplet
        metric_pool = self.config.get("hybrid_metric_pool", self.metric_triplet)
        if not isinstance(metric_pool, (list, tuple)) or not metric_pool:
            metric_pool = self.metric_triplet
        metrics = []
        for metric in metric_pool:
            m = str(metric).lower().strip()
            if m and m not in metrics:
                metrics.append(m)
        for metric in self.metric_triplet:
            if metric not in metrics:
                metrics.append(metric)
        timing_mode, max_ratio, measure_baseline = self._timing_gate_cfg()
        baseline_time = None
        default_baseline = "taylor" if bool(self.config.get("hybrid_measure_taylor_baseline", False)) else m_late
        baseline_method = str(self.config.get("hybrid_timing_baseline_method", default_baseline)).lower().strip()
        if timing_mode != "off" and measure_baseline:
            try:
                t0 = perf_counter()
                _ = self.adapter.get_score_map(model, loader, baseline_method)
                baseline_time = perf_counter() - t0
            except Exception:
                baseline_time = None

        t1 = perf_counter()
        multi_scores = self.adapter.get_multi_metric_scores(model, loader, metrics)
        hybrid_time = perf_counter() - t1

        available_metrics = [str(k).lower().strip() for k in multi_scores.keys()]
        chosen_metrics: List[str] = []
        for m in (m_early, m_mid, m_late):
            if m in available_metrics and m not in chosen_metrics:
                chosen_metrics.append(m)
        fallback_order = ["l1_norm", "mean_abs_act", "apoz", "taylor"]
        for fm in fallback_order:
            if fm in available_metrics and fm not in chosen_metrics:
                chosen_metrics.append(fm)
        for fm in available_metrics:
            if fm not in chosen_metrics:
                chosen_metrics.append(fm)
        if not chosen_metrics:
            raise RuntimeError("HybridMetaPruner received no metric scores from adapter.get_multi_metric_scores().")
        while len(chosen_metrics) < 3:
            chosen_metrics.append(chosen_metrics[-1])
        m_early, m_mid, m_late = chosen_metrics[:3]

        self.timing_report = {"hybrid_time_s": float(hybrid_time)}
        if baseline_time is not None:
            ratio = hybrid_time / max(baseline_time, 1e-12)
            self.timing_report.update(
                {
                    "baseline_method": baseline_method,
                    "baseline_time_s": float(baseline_time),
                    "hybrid_to_baseline_ratio": float(ratio),
                    "max_allowed_ratio": float(max_ratio),
                    # Back-compat aliases for previous reports/tests.
                    "taylor_time_s": float(baseline_time),
                    "hybrid_to_taylor_ratio": float(ratio),
                }
            )
            if ratio > max_ratio:
                msg = (
                    f"Hybrid scoring time ratio {ratio:.2f} exceeded allowed "
                    f"max {max_ratio:.2f}."
                )
                if timing_mode == "error":
                    raise RuntimeError(msg)
                if timing_mode == "warn":
                    print(f"WARNING: {msg}")

        hybrid_score_map = {}
        self.last_metric_weights = {}
        self.last_metric_contributions = {}
        self.last_layer_decisions = {}
        
        for i, layer_name in enumerate(prunable_layers):
            depth = i / (num_layers - 1) if num_layers > 1 else 0

            if self.mode in ("adaptive", "similarity", "cost_aware", "correlation"):
                score, decision = self._adaptive_layer_score(multi_scores, layer_name)
                hybrid_score_map[layer_name] = score
                self.last_layer_decisions[layer_name] = decision
                self.last_metric_weights[layer_name] = decision.get("weights", {})
                self.last_metric_contributions[layer_name] = {
                    k: float(v) for k, v in decision.get("weights", {}).items()
                }
                continue
            
            # Calculate weights based on depth
            w_l1, w_act, w_taylor = self._get_weights(depth)

            s_early = self._score_for_layer(multi_scores, m_early, layer_name)
            s_mid = self._score_for_layer(multi_scores, m_mid, layer_name)
            s_late = self._score_for_layer(multi_scores, m_late, layer_name)

            if bool(self.config.get("hybrid_rank_fusion", True)):
                s_early = self._rank_normalize(s_early)
                s_mid = self._rank_normalize(s_mid)
                s_late = self._rank_normalize(s_late)
            else:
                s_early = self._normalize(s_early)
                s_mid = self._normalize(s_mid)
                s_late = self._normalize(s_late)

            w_l1, w_act, w_taylor = self._confidence_adjusted_weights(
                (w_l1, w_act, w_taylor),
                (s_early, s_mid, s_late),
            )
            
            # Weighted ensemble
            s_total = (w_l1 * s_early) + (w_act * s_mid) + (w_taylor * s_late)

            self.last_metric_weights[layer_name] = {
                m_early: float(w_l1),
                m_mid: float(w_act),
                m_late: float(w_taylor),
            }
            self.last_metric_contributions[layer_name] = {
                m_early: float((w_l1 * s_early).mean()),
                m_mid: float((w_act * s_mid).mean()),
                m_late: float((w_taylor * s_late).mean()),
            }
            
            # 3. Conflict Resolution (Safety First)
            # Protect top 5% of any individual metric
            protection_mask = (
                (s_early > np.percentile(s_early, 95))
                | (s_mid > np.percentile(s_mid, 95))
                | (s_late > np.percentile(s_late, 95))
            )
            
            # Boost scores of protected filters to ensure they are kept
            s_total[protection_mask] = np.maximum(s_total[protection_mask], 1.0)
            
            hybrid_score_map[layer_name] = s_total
            
        return hybrid_score_map

    def _get_weights(self, depth: float) -> Tuple[float, float, float]:
        """Calculates metric weights for a given relative depth [0, 1]."""
        if self.mode == 'bucket':
            if depth < 0.25: return 1.0, 0.0, 0.0
            if depth < 0.75: return 0.0, 1.0, 0.0
            return 0.0, 0.0, 1.0
        
        # Smooth blending (Linear Interpolation)
        # Zone 1 (0 - 0.25): transition L1 -> Act
        # Zone 2 (0.25 - 0.75): transition Act -> Taylor
        # Zone 3 (0.75 - 1.0): Taylor
        
        if depth < 0.25:
            # depth 0: 100% L1, depth 0.25: 100% Act
            alpha = depth / 0.25
            return (1.0 - alpha), alpha, 0.0
        elif depth < 0.75:
            # depth 0.25: 100% Act, depth 0.75: 100% Taylor
            alpha = (depth - 0.25) / 0.5
            return 0.0, (1.0 - alpha), alpha
        else:
            return 0.0, 0.0, 1.0

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-Max normalization to [0, 1] for ensemble consistency."""
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.size == 0:
            return scores
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            return (scores - s_min) / (s_max - s_min)
        return np.ones_like(scores)

    def _rank_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Converts arbitrary scores to percentile ranks in [0, 1]."""
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        if s.size <= 1:
            return np.ones_like(s)
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(s.size, dtype=np.float64)
        return ranks / float(max(s.size - 1, 1))

    def _confidence_adjusted_weights(
        self,
        weights: Tuple[float, float, float],
        scores: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> Tuple[float, float, float]:
        """Downweights flat or low-information metrics before blending."""
        if not bool(self.config.get("hybrid_confidence_weighting", True)):
            return weights
        conf = []
        for s in scores:
            arr = np.asarray(s, dtype=np.float64).reshape(-1)
            if arr.size <= 1:
                conf.append(1.0)
            else:
                conf.append(float(np.std(arr)) + 1e-6)
        adjusted = np.asarray(weights, dtype=np.float64) * np.asarray(conf, dtype=np.float64)
        total = float(np.sum(adjusted))
        if total <= 0:
            return weights
        adjusted = adjusted / total
        return float(adjusted[0]), float(adjusted[1]), float(adjusted[2])

    def _adaptive_layer_score(
        self,
        scores: Dict[str, Dict[str, np.ndarray]],
        layer_name: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Selects or blends methods by rank correlation, confidence, and cost.

        If a cheap method is highly rank-correlated with one or more expensive
        methods in the current layer, the cheap proxy is selected. Otherwise the
        layer falls back to a cost-aware weighted blend of available methods.
        """
        threshold = float(self.config.get("hybrid_correlation_threshold", 0.90))
        overlap_threshold = float(self.config.get("hybrid_topk_overlap_threshold", 0.80))
        min_conf = float(self.config.get("hybrid_min_confidence", 1e-6))
        vectors: Dict[str, np.ndarray] = {}
        confidence: Dict[str, float] = {}

        target_size = None
        for method, layer_scores in scores.items():
            if layer_name not in layer_scores:
                continue
            raw = np.asarray(layer_scores[layer_name], dtype=np.float64).reshape(-1)
            if raw.size == 0:
                continue
            if target_size is None:
                target_size = raw.size
            if raw.size != target_size:
                continue
            ranked = self._rank_normalize(raw) if bool(self.config.get("hybrid_rank_fusion", True)) else self._normalize(raw)
            vectors[str(method).lower().strip()] = ranked
            confidence[str(method).lower().strip()] = float(np.std(ranked))

        if not vectors:
            return np.ones((1,), dtype=np.float64), {
                "mode": "fallback_empty",
                "selected": "constant",
                "weights": {"constant": 1.0},
                "correlations": {},
            }

        methods = sorted(vectors, key=lambda m: (self._method_cost(m), m))
        simple_methods = {m for m in methods if self._is_simple_method(m)}
        complex_methods = [m for m in methods if m not in simple_methods]
        correlations: Dict[str, Dict[str, float]] = {}
        for a in methods:
            correlations[a] = {}
            for b in methods:
                if a == b:
                    continue
                correlations[a][b] = self._spearman_corr(vectors[a], vectors[b])

        best_proxy = None
        best_proxy_score = -np.inf
        for cheap in methods:
            if confidence.get(cheap, 0.0) < min_conf:
                continue
            if bool(self.config.get("hybrid_simple_proxy_only", True)) and cheap not in simple_methods:
                continue
            cheap_cost = self._method_cost(cheap)
            covered = [
                expensive
                for expensive in complex_methods
                if self._method_cost(expensive) > cheap_cost
                and abs(correlations[cheap].get(expensive, 0.0)) >= threshold
                and self._topk_prune_overlap(vectors[cheap], vectors[expensive]) >= overlap_threshold
            ]
            if not covered:
                continue
            coverage_value = sum(self._method_cost(m) for m in covered)
            proxy_score = (coverage_value * max(confidence.get(cheap, 0.0), min_conf)) / max(cheap_cost, 1e-12)
            if proxy_score > best_proxy_score:
                best_proxy = (cheap, covered)
                best_proxy_score = proxy_score

        if best_proxy is not None:
            selected, covered = best_proxy
            return vectors[selected], {
                "mode": "cheap_proxy",
                "selected": selected,
                "covered_complex_methods": covered,
                "weights": {selected: 1.0},
                "cost": self._method_cost(selected),
                "correlations": correlations,
                "topk_overlap_threshold": overlap_threshold,
                "correlation_threshold": threshold,
                "similarity_rule": "spearman_rank_correlation_and_prune_set_overlap",
            }

        if not complex_methods and bool(self.config.get("hybrid_allow_simple_only_stack", True)):
            if len(methods) == 1:
                selected = methods[0]
                return vectors[selected], {
                    "mode": "simple_only_single",
                    "selected": selected,
                    "weights": {selected: 1.0},
                    "cost": self._method_cost(selected),
                    "correlations": correlations,
                    "topk_overlap_threshold": overlap_threshold,
                    "correlation_threshold": threshold,
                    "similarity_rule": "single_simple_method_available",
                }

            raw_weights = {}
            for method in methods:
                conf = max(confidence.get(method, 0.0), min_conf)
                raw_weights[method] = conf / max(self._method_cost(method), 1e-12)
            total = sum(raw_weights.values())
            weights = {m: raw_weights[m] / total for m in methods}
            blended = np.zeros_like(next(iter(vectors.values())), dtype=np.float64)
            for method, weight in weights.items():
                blended += float(weight) * vectors[method]
            return blended, {
                "mode": "simple_only_stack",
                "selected": "blend",
                "weights": {m: float(w) for m, w in weights.items()},
                "stack_methods": list(weights.keys()),
                "cost": float(sum(self._method_cost(m) for m in weights)),
                "correlations": correlations,
                "topk_overlap_threshold": overlap_threshold,
                "correlation_threshold": threshold,
                "similarity_rule": "simple_methods_stacked_by_confidence_and_measured_cost",
            }

        if not complex_methods:
            selected = min(methods, key=lambda m: (self._method_cost(m), -confidence.get(m, 0.0), m))
            return vectors[selected], {
                "mode": "simple_only_single",
                "selected": selected,
                "weights": {selected: 1.0},
                "cost": self._method_cost(selected),
                "correlations": correlations,
                "topk_overlap_threshold": overlap_threshold,
                "correlation_threshold": threshold,
                "similarity_rule": "simple_only_stacking_disabled",
            }

        stack_methods = list(complex_methods)
        if bool(self.config.get("hybrid_include_best_simple_representative", True)):
            candidates = [m for m in simple_methods if confidence.get(m, 0.0) >= min_conf]
            if candidates:
                stack_methods.append(min(candidates, key=lambda m: (self._method_cost(m), -confidence.get(m, 0.0), m)))
        if not stack_methods:
            stack_methods = methods[:1]

        raw_weights = {}
        for method in stack_methods:
            conf = max(confidence.get(method, 0.0), min_conf)
            raw_weights[method] = conf / max(self._method_cost(method), 1e-12)
        total = sum(raw_weights.values())
        weights = {m: raw_weights[m] / total for m in stack_methods}
        blended = np.zeros_like(next(iter(vectors.values())), dtype=np.float64)
        for method, weight in weights.items():
            blended += float(weight) * vectors[method]
        return blended, {
            "mode": "cost_aware_blend",
            "selected": "blend",
            "weights": {m: float(w) for m, w in weights.items()},
            "stack_methods": list(weights.keys()),
            "simple_methods": sorted(simple_methods),
            "complex_methods": list(complex_methods),
            "correlations": correlations,
            "topk_overlap_threshold": overlap_threshold,
            "correlation_threshold": threshold,
            "similarity_rule": "spearman_rank_correlation_and_prune_set_overlap",
        }

    def _is_simple_method(self, method: str) -> bool:
        configured_simple = self.config.get("hybrid_simple_methods", None)
        if isinstance(configured_simple, (list, tuple, set)):
            return method in {str(m).lower().strip() for m in configured_simple}
        simple = {
            "l1_norm",
            "l2_norm",
            "custom_l2",
            "mean_abs_act",
            "apoz",
        }
        return method in simple

    def _method_cost(self, method: str) -> float:
        measured_costs = self._load_measured_method_costs()
        if method in measured_costs:
            return float(measured_costs[method])
        costs = {
            "l1_norm": 1.0,
            "l2_norm": 1.2,
            "custom_l2": 1.2,
            "mean_abs_act": 2.0,
            "apoz": 2.0,
            "custom_entropy": 2.5,
            "custom_class_entropy": 3.0,
            "custom_spectral_energy": 3.0,
            "custom_hrank": 3.5,
            "chip": 4.0,
            "custom_reprune": 4.5,
            "taylor": 5.0,
            "custom_tis": 5.0,
            "custom_nisp": 5.5,
            "custom_thinet": 6.0,
            "custom_senpis": 6.5,
        }
        config_costs = self.config.get("hybrid_method_costs", {})
        if isinstance(config_costs, dict) and method in config_costs:
            return float(config_costs[method])
        return float(costs.get(method, 4.0))

    def _load_measured_method_costs(self) -> Dict[str, float]:
        """Loads measured method costs from experiment efficiency JSON records.

        The notebooks write `simplicity_time_sec = prune_time_sec + heal_time_sec`.
        In current outputs, `prune_time_sec` is the overall method cost:
        scoring, mask building, and physical surgery. When supplied through
        `hybrid_efficiency_json_path` or `hybrid_efficiency_records`, these
        measured times become the preferred definition of method simplicity.
        """
        if self._measured_method_costs is not None:
            return self._measured_method_costs

        records = self.config.get("hybrid_efficiency_records", None)
        if records is None:
            path = self.config.get("hybrid_efficiency_json_path", None)
            if path:
                try:
                    with open(Path(path), "r", encoding="utf-8") as f:
                        records = json.load(f)
                except Exception:
                    records = None
        if not isinstance(records, list):
            self._measured_method_costs = {}
            return self._measured_method_costs

        backend_filter = str(self.config.get("backend", "")).lower().strip()
        model_filter = str(self.config.get("model_type", self.config.get("model", ""))).lower().strip()
        dataset_filter = str(self.config.get("dataset", "")).lower().strip()
        grouped: Dict[str, List[float]] = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            if str(rec.get("status", "ok")).lower().strip() not in ("", "ok", "success"):
                continue
            if backend_filter and str(rec.get("backend", "")).lower().strip() not in ("", backend_filter):
                continue
            if model_filter and str(rec.get("model", "")).lower().strip() not in ("", model_filter):
                continue
            if dataset_filter and str(rec.get("dataset", "")).lower().strip() not in ("", dataset_filter):
                continue
            method = str(rec.get("method", "")).lower().strip()
            if not method:
                continue
            value = rec.get("simplicity_time_sec", None)
            if value is None:
                value = float(rec.get("prune_time_sec", 0.0) or 0.0) + float(rec.get("heal_time_sec", 0.0) or 0.0)
            try:
                cost = float(value)
            except Exception:
                continue
            if np.isfinite(cost) and cost > 0:
                grouped.setdefault(method, []).append(cost)

        costs = {m: float(np.median(v)) for m, v in grouped.items() if v}
        if costs and bool(self.config.get("hybrid_normalize_measured_costs", True)):
            floor = min(v for v in costs.values() if v > 0)
            costs = {m: float(v / floor) for m, v in costs.items()}
        self._measured_method_costs = costs
        return self._measured_method_costs

    def _spearman_corr(self, a: np.ndarray, b: np.ndarray) -> float:
        ra = self._rank_normalize(a)
        rb = self._rank_normalize(b)
        if ra.size != rb.size or ra.size <= 1:
            return 0.0
        if float(np.std(ra)) < 1e-12 or float(np.std(rb)) < 1e-12:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])

    def _topk_prune_overlap(self, a: np.ndarray, b: np.ndarray) -> float:
        """Agreement between the channels each method would prune.

        Lower scores are less important, so the compared set is the bottom-k
        channels under each rank vector. This mirrors the pruning-literature
        notion of similar criteria producing similar selected filter indices.
        """
        va = np.asarray(a, dtype=np.float64).reshape(-1)
        vb = np.asarray(b, dtype=np.float64).reshape(-1)
        if va.size != vb.size or va.size == 0:
            return 0.0
        ratio = float(self.config.get("current_prune_ratio", self.config.get("ratio", 0.30)))
        ratio = float(np.clip(ratio, 0.0, 0.95))
        k = max(1, int(round(va.size * ratio)))
        idx_a = set(np.argsort(va, kind="mergesort")[:k].tolist())
        idx_b = set(np.argsort(vb, kind="mergesort")[:k].tolist())
        return float(len(idx_a & idx_b) / max(k, 1))

    def _score_for_layer(self, scores: Dict[str, Dict[str, np.ndarray]], method: str, layer_name: str) -> np.ndarray:
        method_scores = scores.get(method, {})
        if layer_name in method_scores:
            return np.asarray(method_scores[layer_name], dtype=np.float64).reshape(-1)
        if method_scores:
            # Last-resort shape fallback keeps hybrid diagnostics robust when a
            # registered custom method skips a layer.
            first = next(iter(method_scores.values()))
            return np.ones_like(np.asarray(first, dtype=np.float64).reshape(-1))
        return np.ones((1,), dtype=np.float64)
