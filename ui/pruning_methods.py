"""Runtime pruning-method registrations used by the ReduCNN UI."""

from __future__ import annotations

import numpy as np

from reducnn.pruner.registry import list_method_names, register_method


METHOD_CATALOG = {
    "l1_norm": {
        "label": "L1 norm",
        "description": "Weight-magnitude pruning. Fast and useful as a stable baseline.",
    },
    "mean_abs_act": {
        "label": "Mean absolute activation",
        "description": "Keeps channels with strong calibration activations.",
    },
    "apoz": {
        "label": "APoZ",
        "description": "Activation Percentage of Zeros, implemented as a keep score.",
    },
    "chip": {
        "label": "CHIP",
        "description": "Channel-independence score based on activation correlation.",
    },
    "custom_l2": {
        "label": "L2 norm",
        "description": "Weight-energy pruning using per-channel L2 scores.",
    },
    "custom_entropy": {
        "label": "Activation entropy",
        "description": "Ranks channels by activation distribution entropy.",
    },
    "custom_class_entropy": {
        "label": "Class entropy",
        "description": "Favors channels whose responses are class-discriminative.",
    },
    "custom_hrank": {
        "label": "HRank",
        "description": "Uses feature-map matrix rank as a channel importance proxy.",
    },
    "custom_spectral_energy": {
        "label": "Spectral energy",
        "description": "Ranks channels by frequency-domain activation energy.",
    },
    "custom_senpis": {
        "label": "SENPIS",
        "description": "Combines sensitivity, entropy, and weight energy signals.",
    },
    "custom_tis": {
        "label": "TIS",
        "description": "Thresholded class-wise Taylor importance scoring.",
    },
    "custom_nisp": {
        "label": "NISP",
        "description": "Propagates downstream importance back through prunable layers.",
    },
    "custom_thinet": {
        "label": "ThiNet-style",
        "description": "Uses next-layer reconstruction influence and activations.",
    },
    "custom_reprune": {
        "label": "RePrune-style",
        "description": "Keeps representative, less redundant activation channels.",
    },
    "hybrid": {
        "label": "Hybrid meta-pruner",
        "description": "Depth-aware ensemble over available scoring methods.",
    },
}

def register_ui_methods() -> None:
    """Register helper-backed pruning methods for both PyTorch and Keras."""

    @register_method("custom_l2", framework="global")
    @register_method("l2_norm", framework="global")
    def _l2(layer, tools=None, **kwargs):
        return tools.weight_l2(layer) if tools is not None else None

    @register_method("chip", framework="global")
    def _chip(layer, tools=None, **kwargs):
        if tools is None:
            return None
        act, _ = tools.collect_layer_outputs(layer, include_labels=False)
        return tools.chip_scores(act) if act is not None else tools.weight_l2(layer)

    @register_method("custom_entropy", framework="global")
    def _entropy(layer, tools=None, **kwargs):
        if tools is None:
            return None
        act, _ = tools.collect_layer_outputs(layer, include_labels=False)
        if act is None:
            return tools.weight_l2(layer)
        return np.asarray([tools.entropy_1d(row) for row in tools.channel_matrix(act)], dtype=np.float64)

    @register_method("custom_class_entropy", framework="global")
    def _class_entropy(layer, tools=None, **kwargs):
        if tools is None:
            return None
        act, labels = tools.collect_layer_outputs(layer, include_labels=True)
        if act is None or labels is None:
            return tools.weight_l2(layer)
        pooled = tools.pooled_nc(act)
        classes = sorted(set(int(x) for x in np.asarray(labels).reshape(-1)))
        if not classes:
            return tools.weight_l2(layer)
        mat = []
        for cls in classes:
            idx = np.asarray(labels).reshape(-1) == cls
            if np.any(idx):
                mat.append(np.mean(np.abs(pooled[idx]), axis=0))
        return tools.class_entropy_discriminability(np.asarray(mat)) if mat else tools.weight_l2(layer)

    @register_method("custom_hrank", framework="global")
    def _hrank(layer, tools=None, **kwargs):
        if tools is None:
            return None
        act, _ = tools.collect_layer_outputs(layer, include_labels=False)
        return tools.rank_scores(act) if act is not None else tools.weight_l2(layer)

    @register_method("custom_spectral_energy", framework="global")
    def _spectral(layer, tools=None, **kwargs):
        if tools is None:
            return None
        act, _ = tools.collect_layer_outputs(layer, include_labels=False)
        return tools.spectral_energy_scores(act) if act is not None else tools.weight_l2(layer)

    @register_method("custom_senpis", framework="global")
    def _senpis(layer, tools=None, **kwargs):
        if tools is None:
            return None
        return tools.senpis_ablation_scores(
            layer,
            similarity_threshold=float(kwargs.get("senpis_similarity_threshold", 0.90)),
            attenuation_factor=float(kwargs.get("senpis_attenuation_factor", 0.5)),
        )

    @register_method("custom_tis", framework="global")
    def _tis(layer, tools=None, **kwargs):
        if tools is None:
            return None
        mat = tools.classwise_taylor_matrix(layer)
        if mat is not None:
            return tools.tis_threshold_aggregate(mat, percentile=float(kwargs.get("tis_percentile", 75.0)))
        act, _ = tools.collect_layer_outputs(layer, include_labels=False)
        return tools.rank_scores(act) if act is not None else tools.weight_l2(layer)

    @register_method("custom_nisp", framework="global")
    def _nisp(layer_name=None, tools=None, **kwargs):
        if tools is None:
            return None
        scores = tools.nisp_score_map()
        return scores.get(layer_name)

    @register_method("custom_thinet", framework="global")
    def _thinet(layer, tools=None, **kwargs):
        if tools is None:
            return None
        return tools.thinet_next_layer_damage_scores(layer)

    @register_method("custom_reprune", framework="global")
    def _reprune(layer, tools=None, **kwargs):
        if tools is None:
            return None
        scores = tools.reprune_kernel_coverage_scores(
            layer,
            target_keep_ratio=float(kwargs.get("reprune_target_keep_ratio", 1.0 - float(kwargs.get("ratio", 0.3)))),
        )
        if scores is not None:
            return scores
        act, _ = tools.collect_layer_outputs(layer, include_labels=False)
        return tools.reprune_representative_scores(act) if act is not None else tools.weight_l2(layer)


def get_method_catalog(framework: str = "torch") -> dict:
    """Returns UI metadata for every method currently visible in the registry."""
    register_ui_methods()
    names = list_method_names(framework, include_global=True)
    catalog = {}
    for name in names:
        catalog[name] = METHOD_CATALOG.get(
            name,
            {
                "label": name.replace("_", " ").title(),
                "description": "Registered custom pruning method.",
            },
        )
    if "hybrid" not in catalog:
        catalog["hybrid"] = METHOD_CATALOG["hybrid"]
    return dict(sorted(catalog.items()))
