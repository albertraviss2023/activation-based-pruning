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
        triplet = self.config.get("hybrid_metric_triplet", ["l1_norm", "mean_abs_act", "apoz"])
        if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
            triplet = ["l1_norm", "mean_abs_act", "apoz"]
        self.metric_triplet = [str(m).lower().strip() for m in triplet]

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
        metrics = [m_early, m_mid, m_late]
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
        
        for i, layer_name in enumerate(prunable_layers):
            depth = i / (num_layers - 1) if num_layers > 1 else 0
            
            # Calculate weights based on depth
            w_l1, w_act, w_taylor = self._get_weights(depth)

            s_early = self._normalize(multi_scores[m_early][layer_name])
            s_mid = self._normalize(multi_scores[m_mid][layer_name])
            s_late = self._normalize(multi_scores[m_late][layer_name])
            
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
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            return (scores - s_min) / (s_max - s_min)
        return np.ones_like(scores)
