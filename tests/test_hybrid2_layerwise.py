from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from reducnn.pruner.hybrid2 import choose_layerwise_overlap_representatives, prune_set_overlap


def test_prune_set_overlap_uses_bottom_pruned_filters():
    a = np.asarray([0.1, 0.2, 0.8, 0.9])
    b = np.asarray([0.0, 0.3, 0.7, 0.6])
    c = np.asarray([0.8, 0.9, 0.1, 0.2])

    assert prune_set_overlap(a, b, ratio=0.5) == 1.0
    assert prune_set_overlap(a, c, ratio=0.5) == 0.0


def test_hybrid2_selects_fast_representative_from_overlap_group():
    score_maps = {
        "cheap": {"conv": np.asarray([0.1, 0.2, 0.8, 0.9])},
        "slow": {"conv": np.asarray([0.0, 0.3, 0.7, 0.6])},
        "other": {"conv": np.asarray([0.8, 0.9, 0.1, 0.2])},
    }
    evidence = [
        {"method": "cheap", "total_time_sec": 1.0, "flops_reduction_pct": 40.0, "simplicity_rank": 1},
        {"method": "slow", "total_time_sec": 4.0, "flops_reduction_pct": 42.0, "simplicity_rank": 3},
        {"method": "other", "total_time_sec": 1.5, "flops_reduction_pct": 80.0, "simplicity_rank": 2},
    ]

    selected, decisions, pairs = choose_layerwise_overlap_representatives(
        score_maps,
        ratio=0.5,
        evidence=evidence,
        overlap_threshold=0.8,
    )

    assert set(selected) == {"conv"}
    assert decisions[0]["mode"] == "overlap_representative"
    assert decisions[0]["chosen_method"] == "cheap"
    assert decisions[0]["agreement_group"] == ["cheap", "slow"]
    assert any(p["passes_agreement"] for p in pairs)


def test_hybrid2_flags_no_overlap_instead_of_claiming_stack():
    score_maps = {
        "a": {"conv": np.asarray([0.1, 0.2, 0.8, 0.9])},
        "b": {"conv": np.asarray([0.8, 0.9, 0.1, 0.2])},
    }
    evidence = [
        {"method": "a", "total_time_sec": 2.0, "flops_reduction_pct": 40.0, "simplicity_rank": 2},
        {"method": "b", "total_time_sec": 1.0, "flops_reduction_pct": 20.0, "simplicity_rank": 1},
    ]

    _, decisions, _ = choose_layerwise_overlap_representatives(
        score_maps,
        ratio=0.5,
        evidence=evidence,
        overlap_threshold=0.8,
    )

    assert decisions[0]["mode"] == "no_overlap_best_evidence"
    assert decisions[0]["agreement_group_size"] == 1
