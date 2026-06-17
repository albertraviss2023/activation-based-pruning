"""Build the deterministic discovered-stack reproduction notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproduce_discovered_cifar10_global_flops_accuracy_stacks.ipynb"
REFERENCE = ROOT / (
    "experiments_for_pruning_policy_search_on_context_resnet18_cifar10_"
    "registered_methods_objective_flops_accuracy.ipynb"
)


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def reference_registration_cell() -> str:
    """Reuse the exact custom-method registrations used during discovery."""
    notebook = json.loads(REFERENCE.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][12]["source"])
    source = source.replace("# [ReduCNN LFPC cell note]\n", "", 1)
    source = source.replace(
        'CALIB_BATCHES = int(globals().get("SCORING_CALIB_BATCHES", globals().get("CALIB_BATCHES", 2)))\n'
        'BINS = int(globals().get("BINS", 32))\n'
        'EPS = float(globals().get("EPS", 1e-12))',
        '''def _registration_default(value, default):
    return default if value is None or value == "" else value


CALIB_BATCHES = int(_registration_default(globals().get("SCORING_CALIB_BATCHES", globals().get("CALIB_BATCHES", 2)), 2))
BINS = int(_registration_default(globals().get("BINS", 32), 32))
EPS = float(_registration_default(globals().get("EPS", 1e-12), 1e-12))''',
    )
    source = source.replace(
        'return int(kwargs.get("calib_batches", kwargs.get("prune_batches", CALIB_BATCHES)))',
        '''value = kwargs.get("calib_batches", kwargs.get("prune_batches", CALIB_BATCHES))
    return int(_registration_default(value, CALIB_BATCHES))''',
    )
    old_chip = '''def chip_score(layer, **kwargs):
    """CHIP-style channel independence score (paper-aligned correlation form)."""
    from reducnn.pruner.chip import chip_channel_independence_scores

    tools = _tools(kwargs)
    A, _ = tools.collect_layer_outputs(layer, max_batches=_max_batches(kwargs), include_labels=False)
    if A is None:
        return None

    channel_axis = 1 if tools.framework == "torch" else -1
    spatial_total = int(A.shape[2] * A.shape[3]) if tools.framework == "torch" else int(A.shape[1] * A.shape[2])
    max_spatial = int(kwargs.get("chip_max_spatial", spatial_total))
    max_spatial = max(1, min(max_spatial, spatial_total))

    s = chip_channel_independence_scores(
        A,
        channel_axis=channel_axis,
        max_spatial=max_spatial,
    )
    return np.asarray(s, dtype=np.float64).reshape(-1)
'''
    rank_safe_chip = '''def chip_score(layer, **kwargs):
    """CHIP channel-independence score with spatial and dense fallbacks."""
    tools = _tools(kwargs)
    A, _ = tools.collect_layer_outputs(
        layer,
        max_batches=_max_batches(kwargs),
        include_labels=False,
    )
    if A is None:
        return None

    # Adapter scoring can visit both convolutional and dense-like layers.
    # CustomMethodTools.chip_scores uses the paper-aligned spatial score for
    # rank-4 activations and a channel-independence fallback for other ranks.
    max_spatial = kwargs.get("chip_max_spatial")
    if np.asarray(A).ndim == 4 and max_spatial is not None:
        max_spatial = max(1, int(max_spatial))
    else:
        max_spatial = None
    scores = tools.chip_scores(A, max_spatial=max_spatial)
    return np.asarray(scores, dtype=np.float64).reshape(-1)
'''
    if old_chip not in source:
        raise RuntimeError(
            "The reference CHIP registration changed; update the rank-safe "
            "replacement in this notebook builder."
        )
    return source.replace(old_chip, rank_safe_chip, 1)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3"}
    nb["cells"] = [
        markdown(
            """
# Reproduce the discovered CIFAR-10 global FLOPs-accuracy stacks

This notebook applies six **frozen, already-discovered layer-wise policies**. It
does not train LFPC, search thresholds, rank candidate policies, or select a new
method. For every stack it:

1. loads the same CIFAR-10 preprocessing and saved baseline used by the objective notebooks;
2. scores only the pruning methods referenced by that model's frozen policies;
3. assigns each prunable layer its recorded method;
4. builds one global structural mask at the declared pruning ratio;
5. performs ReduCNN structural surgery, optional healing, validation/test evaluation, and profiling;
6. saves the structurally pruned model plus a complete provenance record.

The four-digit thesis IDs are preserved. Stack `9738` also records the latest
report-artifact alias `9245`, so either identifier can be traced without silently
changing the thesis figure.
"""
        ),
        markdown(
            """
## Colab bootloader and ReduCNN installation

This cell prepares a fresh runtime before any project imports. In Google Colab it
mounts Drive, locates the repository (including Drive shortcuts), changes into
the repository, installs ReduCNN and its declared dependencies in editable mode,
checks PyTorch/torchvision, and verifies the loaded package path. Locally it
locates the current checkout without reinstalling it unless explicitly requested.

For a nonstandard Drive location, set:

```python
os.environ["REDUCNN_PROJECT_PATH"] = "/content/drive/MyDrive/path/to/activation-based-pruning"
```
"""
        ),
        code(
            r'''
import importlib
import os
import subprocess
import sys
from pathlib import Path


def is_reducnn_repository(path):
    path = Path(path)
    return (
        (path / "src" / "reducnn").is_dir()
        and (path / "pyproject.toml").is_file()
    )


def locate_local_repository(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if is_reducnn_repository(candidate):
            return candidate
    return None


def locate_colab_repository():
    override = os.environ.get("REDUCNN_PROJECT_PATH", "").strip()
    direct_candidates = [
        override,
        "/content/drive/Othercomputers/My laptop/activation-based-pruning",
        "/content/drive/Othercomputers/My laptop (1)/activation-based-pruning",
        "/content/drive/MyDrive/activation-based-pruning",
        "/content/drive/MyDrive/activation-based-prunning",
        "/content/drive/MyDrive/Shared with me/activation-based-pruning",
        "/content/drive/Shared with me/activation-based-pruning",
    ]
    for candidate in direct_candidates:
        if candidate and is_reducnn_repository(candidate):
            return Path(candidate).resolve()

    search_roots = [
        Path("/content/drive/.shortcut-targets-by-id"),
        Path("/content/drive/Othercomputers"),
        Path("/content/drive/MyDrive"),
        Path("/content/drive/Shared with me"),
    ]
    names = {"activation-based-pruning", "activation-based-prunning"}
    skipped = {".git", ".pytest_cache", "__pycache__", "node_modules", "outputs"}
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for current, directories, _files in os.walk(search_root):
            directories[:] = [name for name in directories if name not in skipped]
            for name in list(directories):
                if name in names:
                    candidate = Path(current) / name
                    if is_reducnn_repository(candidate):
                        return candidate.resolve()
    return None


try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    drive = None
    IN_COLAB = False

if IN_COLAB:
    drive.mount("/content/drive", force_remount=False)
    PROJECT_ROOT = locate_colab_repository()
else:
    PROJECT_ROOT = locate_local_repository()

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Could not locate the ReduCNN repository. In Colab, add the repository "
        "to Drive or set REDUCNN_PROJECT_PATH to its exact directory."
    )

os.chdir(PROJECT_ROOT)
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "ui"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

INSTALL_REDUCNN_PACKAGE = bool(globals().get("INSTALL_REDUCNN_PACKAGE", IN_COLAB))
BOOT_DRY_RUN = os.environ.get("REDUCNN_REPRO_DRY_RUN", "0") == "1"
if INSTALL_REDUCNN_PACKAGE:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
         "pip", "setuptools", "wheel"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT_ROOT)],
        check=True,
    )

if BOOT_DRY_RUN:
    torch = torchvision = None
else:
    try:
        import torch
        import torchvision
    except ImportError:
        if not IN_COLAB:
            raise ImportError(
                "PyTorch and torchvision are required for a real pruning run."
            )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"],
            check=True,
        )
        import torch
        import torchvision

reducnn_spec = importlib.util.find_spec("reducnn")
if reducnn_spec is None:
    raise ImportError("ReduCNN is not importable after environment preparation.")

if IN_COLAB or INSTALL_REDUCNN_PACKAGE:
    import reducnn
    reducnn_version = getattr(reducnn, "__version__", "unknown")
    reducnn_location = reducnn.__file__
else:
    reducnn_version = "not imported during local boot smoke"
    reducnn_location = reducnn_spec.origin

print("Runtime:", "Google Colab" if IN_COLAB else "local")
print("Project root:", PROJECT_ROOT)
print("ReduCNN version:", reducnn_version)
print("ReduCNN module:", reducnn_location)
if BOOT_DRY_RUN:
    print("PyTorch/torchvision check: skipped for dry-run validation")
else:
    print("PyTorch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
'''
        ),
        markdown(
            """
## Runtime controls

Set `STACK_IDS_TO_RUN` to a subset for a partial run. Existing successful
checkpoints are reused unless `FORCE_REPRUNE_DISCOVERED_STACKS=True`.
`REPRODUCTION_DRY_RUN=True` validates policies and renders timelines without
loading data or pruning models.

By default, the notebook reproduces exactly three frozen stacks for each model:
ratios `0.30`, `0.45`, and `0.55`. Set `MODELS_TO_RUN` to `["vgg16"]` or
`["resnet18"]` to run one architecture while retaining all three stacks.
"""
        ),
        code(
            r'''
import copy
import json
import os
import random
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "reducnn").exists():
            return candidate
    raise FileNotFoundError("Could not locate a repository containing src/reducnn.")


ROOT = Path(PROJECT_ROOT).resolve() if "PROJECT_ROOT" in globals() else find_project_root()
for path in (ROOT, ROOT / "src", ROOT / "ui"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SEED = int(globals().get("SEED", 42))
random.seed(SEED)
np.random.seed(SEED)

REPRODUCTION_DRY_RUN = bool(globals().get(
    "REPRODUCTION_DRY_RUN",
    os.environ.get("REDUCNN_REPRO_DRY_RUN", "0") == "1",
))
FORCE_REPRUNE_DISCOVERED_STACKS = bool(
    globals().get("FORCE_REPRUNE_DISCOVERED_STACKS", False)
)
RUN_HEALING = bool(globals().get("RUN_HEALING", True))
MODELS_TO_RUN = [
    str(value).lower()
    for value in globals().get("MODELS_TO_RUN", ["resnet18", "vgg16"])
]
STACK_IDS_TO_RUN = globals().get(
    "STACK_IDS_TO_RUN",
    None,
)

MANIFEST_PATH = ROOT / "configs" / "discovered_cifar10_global_flops_accuracy_stacks.json"
OUTPUT_ROOT = ROOT / "outputs" / "discovered_stack_reproduction" / "cifar10"
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_ROOT / RUN_STAMP
RUN_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", ROOT)
print("Policy manifest:", MANIFEST_PATH)
print("Output directory:", RUN_DIR)
print("Dry run:", REPRODUCTION_DRY_RUN)
'''
        ),
        markdown(
            """
## Load and validate the frozen policies

This cell checks ID uniqueness, expected layer counts, valid ratios, and duplicate
layer assignments before any expensive work starts. The resulting table is the
run's policy provenance and is exported unchanged.
"""
        ),
        code(
            r'''
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
available_stacks = [
    stack for stack in manifest["stacks"]
    if str(stack["model"]).lower() in MODELS_TO_RUN
]
if STACK_IDS_TO_RUN is None:
    stacks = available_stacks
else:
    selected_ids = {str(value) for value in STACK_IDS_TO_RUN}
    stacks = [
        stack for stack in available_stacks
        if str(stack["stack_id"]) in selected_ids
    ]
if not stacks:
    raise RuntimeError("No selected stack IDs exist in the policy manifest.")

expected_layers = {"resnet18": 20, "vgg16": 13}
expected_ratios = {0.30, 0.45, 0.55}
if STACK_IDS_TO_RUN is None:
    for model_name in MODELS_TO_RUN:
        model_stacks = [
            stack for stack in stacks
            if str(stack["model"]).lower() == model_name
        ]
        model_ratios = {round(float(stack["ratio"]), 2) for stack in model_stacks}
        if len(model_stacks) != 3 or model_ratios != expected_ratios:
            raise RuntimeError(
                f"Expected exactly three frozen {model_name} stacks at ratios "
                f"{sorted(expected_ratios)}, found "
                f"{[(stack['stack_id'], stack['ratio']) for stack in model_stacks]}."
            )
        print(
            f"Selected three {model_name} stacks:",
            [(stack["stack_id"], stack["ratio"]) for stack in model_stacks],
        )
seen_ids = set()
policy_rows = []
for stack in stacks:
    stack_id = str(stack["stack_id"])
    if stack_id in seen_ids:
        raise RuntimeError(f"Duplicate stack ID: {stack_id}")
    seen_ids.add(stack_id)
    if stack["model"] not in expected_layers:
        raise ValueError(f"Unsupported model in manifest: {stack['model']}")
    if len(stack["policy"]) != expected_layers[stack["model"]]:
        raise RuntimeError(
            f"Stack {stack_id} has {len(stack['policy'])} layers; "
            f"expected {expected_layers[stack['model']]}."
        )
    names = [layer for layer, _method in stack["policy"]]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Stack {stack_id} assigns a layer more than once.")
    if not 0.0 < float(stack["ratio"]) < 1.0:
        raise ValueError(f"Invalid ratio for stack {stack_id}: {stack['ratio']}")
    for layer_index, (layer, method) in enumerate(stack["policy"], start=1):
        region = (
            "Early" if layer_index <= int(np.ceil(len(stack["policy"]) / 3))
            else "Middle" if layer_index <= int(np.ceil(2 * len(stack["policy"]) / 3))
            else "Late"
        )
        policy_rows.append({
            "stack_id": stack_id,
            "source_report_stack_id": str(stack.get("source_report_stack_id", stack_id)),
            "source_stack_key": stack["source_stack_key"],
            "dataset": manifest["dataset"],
            "objective": manifest["objective"],
            "scope": manifest["scope"],
            "model": stack["model"],
            "ratio": float(stack["ratio"]),
            "layer_index": layer_index,
            "region": region,
            "layer": layer,
            "selected_method": method,
        })

policy_df = pd.DataFrame(policy_rows)
policy_df.to_csv(RUN_DIR / "frozen_layerwise_policies.csv", index=False)
display(policy_df)
'''
        ),
        markdown(
            """
## Policy timelines

The timelines provide a visual audit before pruning. Early, middle, and late
regions are explicit, and the four-digit stack ID is the same identifier written
into checkpoints and result tables.
"""
        ),
        code(
            r'''
METHOD_LABELS = {
    "chip": "CHIP", "custom_reprune": "REPrune", "custom_nisp": "NISP",
    "custom_senpis": "SeNPIS", "custom_tis": "TIS", "custom_thinet": "ThiNet",
    "custom_gfs": "GFS", "custom_dcp": "DCP", "custom_autodfp": "AutoDFP",
}
METHOD_COLORS = {
    method: color for method, color in zip(
        METHOD_LABELS,
        ["#7C3AED", "#2563EB", "#059669", "#DB2777", "#DC2626",
         "#EA580C", "#0D9488", "#9333EA", "#0891B2"],
    )
}


def plot_policy_timeline(stack, destination):
    rows = policy_df[policy_df["stack_id"] == str(stack["stack_id"])].copy()
    count = len(rows)
    x = np.arange(count)
    fig, ax = plt.subplots(figsize=(max(11, count * 0.58), 3.6))
    boundaries = [0, int(np.ceil(count / 3)), int(np.ceil(2 * count / 3)), count]
    region_colors = ["#E0F2FE", "#F1F5F9", "#FEF3C7"]
    for index, region in enumerate(["Early", "Middle", "Late"]):
        left, right = boundaries[index], boundaries[index + 1]
        ax.axvspan(left - 0.5, right - 0.5, color=region_colors[index], zorder=0)
        ax.text((left + right - 1) / 2, 1.20, region, ha="center", va="center",
                fontsize=10, fontweight="bold")
    for position, row in enumerate(rows.itertuples()):
        method = row.selected_method
        ax.bar(position, 0.72, bottom=0.12, width=0.84,
               color=METHOD_COLORS.get(method, "#64748B"), edgecolor="white")
        ax.text(position, 0.48, METHOD_LABELS.get(method, method),
                ha="center", va="center", rotation=90, color="white",
                fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(rows["layer"], rotation=55, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(-0.6, count - 0.4)
    ax.set_ylim(0, 1.34)
    ax.set_title(
        f"Frozen discovered policy | {stack['model']} | global | "
        f"r={float(stack['ratio']):g} | Stack {stack['stack_id']}",
        fontsize=12, fontweight="bold", pad=20,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.show()
    return destination


timeline_rows = []
for stack in stacks:
    path = RUN_DIR / f"policy_timeline_stack_{stack['stack_id']}.png"
    plot_policy_timeline(stack, path)
    timeline_rows.append({"stack_id": stack["stack_id"], "plot": str(path)})
pd.DataFrame(timeline_rows).to_csv(RUN_DIR / "policy_timeline_index.csv", index=False)
'''
        ),
        markdown(
            """
## Register the pruning methods

The following registrations are copied from the same objective-experiment
implementation that discovered these stacks. ReduCNN remains responsible for
backend-specific activations, gradients, graph surgery, and model profiling.
"""
        ),
        code(
            r'''
if REPRODUCTION_DRY_RUN:
    print("Dry run: skipping ReduCNN backend imports and method registration.")
else:
    from reducnn.backends.factory import get_adapter
    from reducnn.pruner.mask_builder import build_pruning_masks
    try:
        from ui.pruning_methods import register_ui_methods
    except ImportError:
        from pruning_methods import register_ui_methods


    def clean_method_sequence(methods):
        """Return readable method labels for registration diagnostics."""
        return [str(method) for method in methods]


    register_ui_methods()
    print("Base and UI pruning methods registered.")
'''
        ),
        code(
            "if REPRODUCTION_DRY_RUN:\n"
            "    print('Dry run: skipping exact custom-method registrations.')\n"
            "else:\n"
            + textwrap.indent(reference_registration_cell(), "    ")
        ),
        markdown(
            """
## Reproduction helpers

These helpers mirror the objective notebooks' CIFAR-10 split, ImageNet
normalization, baseline loading, score-map orientation, global mask building,
structural surgery, healing, profiling, and checkpoint format.
"""
        ),
        code(
            r'''
if not REPRODUCTION_DRY_RUN:
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    torch = torchvision = transforms = DataLoader = Subset = None

MODEL_CONFIG = {
    "resnet18": {
        "batch_size": 128, "calib_samples": 2000, "policy_samples": 2000,
        "train_samples": 20000,
        "val_samples": 5000, "test_samples": 10000, "scoring_batches": 4,
        "heavy_batches": 2, "finetune_epochs": 2, "max_finetune_batches": 60,
        "chip_max_spatial": 16,
        "baseline": ROOT / "saved_models/baselines/pytorch/cifar-10/resnet18/pytorch_resnet18_cifar-10.pth",
    },
    "vgg16": {
        "batch_size": 64, "calib_samples": 5000, "policy_samples": 5000,
        "train_samples": 35000,
        "val_samples": 5000, "test_samples": 10000, "scoring_batches": 8,
        "heavy_batches": 4, "finetune_epochs": 3, "max_finetune_batches": 80,
        "chip_max_spatial": 32,
        "baseline": ROOT / "saved_models/baselines/pytorch/cifar-10/vgg16/pytorch_vgg16_cifar-10.pth",
    },
}
HEAVY_METHODS = {
    "chip", "custom_senpis", "custom_thinet", "custom_gfs",
    "custom_dcp", "custom_autodfp", "custom_tis", "custom_nisp",
}


def make_cifar10_loaders(cfg):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=str(ROOT / "data"), train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(ROOT / "data"), train=False, download=True, transform=transform
    )
    cursor = 0
    def take(size):
        nonlocal cursor
        end = min(cursor + int(size), len(trainset))
        subset = Subset(trainset, list(range(cursor, end)))
        cursor = end
        return subset
    train = take(cfg["train_samples"])
    calib = take(cfg["calib_samples"])
    policy = take(cfg["policy_samples"])
    val = take(cfg["val_samples"])
    test = Subset(testset, list(range(min(cfg["test_samples"], len(testset)))))
    return {
        "train": DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, num_workers=0),
        "calib": DataLoader(calib, batch_size=cfg["batch_size"], shuffle=False, num_workers=0),
        "policy": DataLoader(policy, batch_size=cfg["batch_size"], shuffle=True, num_workers=0),
        "val": DataLoader(val, batch_size=cfg["batch_size"], shuffle=False, num_workers=0),
        "test": DataLoader(test, batch_size=cfg["batch_size"], shuffle=False, num_workers=0),
    }


def limited_loader(loader, max_batches, shuffle=False):
    max_items = max(1, int(max_batches)) * int(loader.batch_size)
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        subset = Subset(dataset.dataset, list(dataset.indices)[:max_items])
    else:
        subset = Subset(dataset, list(range(min(max_items, len(dataset)))))
    return DataLoader(subset, batch_size=loader.batch_size, shuffle=shuffle, num_workers=0)


def evaluate(adapter, model, loader):
    return float(adapter.evaluate(model, loader))


def profile(adapter, model, loader, label):
    flops, params = adapter.get_stats(model, loader)
    flops, params = float(flops), float(params)
    if not np.isfinite(flops) or flops <= 0:
        raise RuntimeError(f"Invalid FLOPs for {label}: {flops}")
    if not np.isfinite(params) or params <= 0:
        raise RuntimeError(f"Invalid parameter count for {label}: {params}")
    return flops, params


def reduction(base, final):
    return 100.0 * (float(base) - float(final)) / float(base)


def mask_audit(masks):
    rows = []
    for layer, mask in masks.items():
        keep = np.asarray(mask).astype(bool).reshape(-1)
        rows.append({
            "layer": layer,
            "filters": int(keep.size),
            "kept": int(keep.sum()),
            "pruned": int(keep.size - keep.sum()),
        })
    total = sum(row["filters"] for row in rows)
    pruned = sum(row["pruned"] for row in rows)
    return {
        "total_filters": total,
        "total_pruned_filters": pruned,
        "actual_pruned_filter_ratio": pruned / max(total, 1),
        "per_layer": rows,
    }


def score_methods(adapter, baseline_model, calib_loader, methods, cfg):
    score_maps, timings = {}, []
    for method in sorted(methods):
        batches = cfg["heavy_batches"] if method in HEAVY_METHODS else cfg["scoring_batches"]
        loader = limited_loader(calib_loader, batches)
        old_cfg = dict(adapter.config)
        adapter.config["prune_batches"] = batches
        adapter.config["calib_batches"] = batches
        if method == "chip":
            adapter.config["chip_max_spatial"] = min(cfg["chip_max_spatial"], 8)
        started = time.perf_counter()
        try:
            score_maps[method] = adapter.get_score_map(baseline_model, loader, method)
        finally:
            adapter.config.clear()
            adapter.config.update(old_cfg)
        timings.append({
            "method": method,
            "batches": batches,
            "score_time_sec": time.perf_counter() - started,
            "scored_layers": len(score_maps[method]),
        })
        print(f"Scored {METHOD_LABELS.get(method, method)}: {len(score_maps[method])} layers")
    return score_maps, pd.DataFrame(timings)


def policy_score_map(stack, score_maps, model_layer_names):
    selected = OrderedDict()
    missing = []
    for layer, method in stack["policy"]:
        if layer not in model_layer_names:
            missing.append(f"{layer} (not in model)")
        elif method not in score_maps or layer not in score_maps[method]:
            missing.append(f"{layer}/{method} (score absent)")
        else:
            selected[layer] = np.asarray(score_maps[method][layer], dtype=np.float64).reshape(-1)
    if missing:
        raise RuntimeError(
            f"Stack {stack['stack_id']} cannot be reproduced; missing assignments: {missing}"
        )
    return selected


def save_checkpoint(path, model, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model,
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }, path)
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
'''
        ),
        markdown(
            """
## Apply each frozen stack and save its structurally pruned model

Methods are scored once per architecture and reused by its three frozen stacks.
The declared ratio controls one global threshold over the layer-specific score
vectors selected by the timeline. The exported mask audit records the actual
number of filters pruned; FLOPs and parameters are measured from the structurally
pruned model rather than inferred from the requested ratio.
"""
        ),
        code(
            r'''
all_results = []
all_score_timings = []

if REPRODUCTION_DRY_RUN:
    print("Dry run complete: manifest, identifiers, policies, and timelines are valid.")
else:
    for model_name in ["resnet18", "vgg16"]:
        model_stacks = [stack for stack in stacks if stack["model"] == model_name]
        if not model_stacks:
            continue
        cfg = MODEL_CONFIG[model_name]
        if not cfg["baseline"].exists():
            raise FileNotFoundError(f"Required baseline checkpoint is missing: {cfg['baseline']}")

        print(f"\nPreparing {model_name}...")
        loaders = make_cifar10_loaders(cfg)
        adapter_cfg = {
            "backend": "pytorch", "dataset": "cifar-10", "model_type": model_name,
            "input_shape": (3, 32, 32), "num_classes": 10,
            "prune_batches": cfg["scoring_batches"],
            "calib_batches": cfg["scoring_batches"],
            "chip_max_spatial": cfg["chip_max_spatial"],
            "lr": 1e-4, "torch_restore_best": True,
        }
        adapter = get_adapter(None, adapter_cfg)
        baseline = adapter.get_model(
            model_name, input_shape=(3, 32, 32), num_classes=10, pretrained=False
        )
        adapter.load_checkpoint(baseline, str(cfg["baseline"]))
        baseline = baseline.to(DEVICE)
        baseline_val = evaluate(adapter, baseline, loaders["val"])
        baseline_test = evaluate(adapter, baseline, loaders["test"])
        baseline_flops, baseline_params = profile(adapter, baseline, loaders["test"], model_name)
        print(
            f"{model_name} baseline: val={baseline_val:.2f}%, test={baseline_test:.2f}%, "
            f"FLOPs={baseline_flops:,.0f}, params={baseline_params:,.0f}"
        )

        required_methods = {
            method for stack in model_stacks for _layer, method in stack["policy"]
        }
        score_maps, score_timing = score_methods(
            adapter, baseline, loaders["calib"], required_methods, cfg
        )
        score_timing["model"] = model_name
        all_score_timings.extend(score_timing.to_dict(orient="records"))
        model_layer_names = set(dict(baseline.named_modules()))

        for stack in model_stacks:
            stack_id = str(stack["stack_id"])
            stack_dir = OUTPUT_ROOT / model_name / f"stack_{stack_id}"
            checkpoint = stack_dir / f"stack_{stack_id}_global_r{stack['ratio']:g}.pt"
            metrics_path = stack_dir / f"stack_{stack_id}_metrics.json"
            if checkpoint.exists() and metrics_path.exists() and not FORCE_REPRUNE_DISCOVERED_STACKS:
                cached = json.loads(metrics_path.read_text(encoding="utf-8"))
                cached["cache_status"] = "reused"
                all_results.append(cached)
                print(f"Reused stack {stack_id}: {checkpoint}")
                continue

            print(f"\nReproducing stack {stack_id} ({model_name}, r={stack['ratio']})")
            selected_scores = policy_score_map(stack, score_maps, model_layer_names)
            started = time.perf_counter()
            masks = build_pruning_masks(
                selected_scores, ratio=float(stack["ratio"]), scope="global"
            )
            mask_build_time = time.perf_counter() - started
            audit = mask_audit(masks)

            started = time.perf_counter()
            pruned_model = adapter.apply_surgery(copy.deepcopy(baseline), masks)
            surgery_time = time.perf_counter() - started

            raw_val = evaluate(adapter, pruned_model, loaders["val"])
            raw_test = evaluate(adapter, pruned_model, loaders["test"])
            raw_flops, raw_params = profile(
                adapter, pruned_model, loaders["test"], f"stack {stack_id} raw"
            )

            heal_time = 0.0
            if RUN_HEALING and cfg["finetune_epochs"] > 0:
                started = time.perf_counter()
                adapter.train(
                    pruned_model,
                    limited_loader(
                        loaders["train"], cfg["max_finetune_batches"], shuffle=True
                    ),
                    cfg["finetune_epochs"],
                    name=f"reproduce_stack_{stack_id}",
                    val_loader=loaders["val"],
                    plot=False,
                )
                heal_time = time.perf_counter() - started

            final_val = evaluate(adapter, pruned_model, loaders["val"])
            final_test = evaluate(adapter, pruned_model, loaders["test"])
            final_flops, final_params = profile(
                adapter, pruned_model, loaders["test"], f"stack {stack_id} final"
            )
            unique_methods = list(dict.fromkeys(method for _layer, method in stack["policy"]))
            scoring_time = float(score_timing[
                score_timing["method"].isin(unique_methods)
            ]["score_time_sec"].sum())

            metadata = {
                "schema_version": "1.0",
                "artifact_type": "frozen_discovered_hybrid_stack",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "stack_id": stack_id,
                "source_report_stack_id": str(stack.get("source_report_stack_id", stack_id)),
                "source_stack_key": stack["source_stack_key"],
                "source_run_id": stack["source_run_id"],
                "dataset": "cifar10",
                "model": model_name,
                "objective": "flops_accuracy",
                "objective_label": "FLOPs + Accuracy",
                "scope": "global",
                "ratio": float(stack["ratio"]),
                "layer_policy": [
                    {"layer_index": index, "layer": layer, "selected_method": method}
                    for index, (layer, method) in enumerate(stack["policy"], start=1)
                ],
                "unique_selected_methods": unique_methods,
                "baseline_checkpoint": str(cfg["baseline"]),
                "baseline_validation_accuracy_pct": baseline_val,
                "baseline_test_accuracy_pct": baseline_test,
                "baseline_flops": baseline_flops,
                "baseline_params": baseline_params,
                "raw_validation_accuracy_pct": raw_val,
                "raw_test_accuracy_pct": raw_test,
                "raw_flops": raw_flops,
                "raw_params": raw_params,
                "final_validation_accuracy_pct": final_val,
                "final_test_accuracy_pct": final_test,
                "accuracy_delta_pp": final_test - baseline_test,
                "final_flops": final_flops,
                "final_params": final_params,
                "flops_reduction_pct": reduction(baseline_flops, final_flops),
                "params_reduction_pct": reduction(baseline_params, final_params),
                "selected_method_scoring_time_sec": scoring_time,
                "mask_build_time_sec": mask_build_time,
                "structural_surgery_time_sec": surgery_time,
                "healing_time_sec": heal_time,
                "deployment_pruning_time_sec": (
                    scoring_time + mask_build_time + surgery_time + heal_time
                ),
                "mask_audit": audit,
                "checkpoint_path": str(checkpoint),
                "cache_status": "computed",
            }
            save_checkpoint(checkpoint, pruned_model, metadata)
            stack_dir.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8"
            )
            all_results.append(metadata)
            print(
                f"Saved stack {stack_id}: test={final_test:.2f}%, "
                f"FLOPs reduction={metadata['flops_reduction_pct']:.2f}%, "
                f"params reduction={metadata['params_reduction_pct']:.2f}%"
            )

results_df = pd.DataFrame(all_results)
timing_df = pd.DataFrame(all_score_timings)
results_df.to_csv(RUN_DIR / "reproduced_stack_metrics.csv", index=False)
timing_df.to_csv(RUN_DIR / "method_score_timing.csv", index=False)
display(results_df)
'''
        ),
        markdown(
            """
## Final run manifest

The final manifest links every requested thesis ID to its source policy,
checkpoint, metrics, and timeline. Reporting code can consume this file without
guessing which experiment context produced a model.
"""
        ),
        code(
            r'''
run_manifest = {
    "schema_version": "1.0",
    "run_id": f"discovered_stack_reproduction_{RUN_STAMP}",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset": "cifar10",
    "objective": "flops_accuracy",
    "scope": "global",
    "policy_manifest": str(MANIFEST_PATH),
    "stack_ids_requested": [str(stack["stack_id"]) for stack in stacks],
    "models_requested": MODELS_TO_RUN,
    "dry_run": REPRODUCTION_DRY_RUN,
    "force_reprune": FORCE_REPRUNE_DISCOVERED_STACKS,
    "run_healing": RUN_HEALING,
    "output_directory": str(RUN_DIR),
    "result_rows": results_df.to_dict(orient="records"),
    "policy_timelines": timeline_rows,
}
(RUN_DIR / "run_manifest.json").write_text(
    json.dumps(run_manifest, indent=2, default=str), encoding="utf-8"
)
(RUN_DIR / "run_manifest.txt").write_text(
    "\n".join([
        f"run_id: {run_manifest['run_id']}",
        "dataset: cifar10",
        "objective: flops_accuracy",
        "scope: global",
        f"models: {', '.join(MODELS_TO_RUN)}",
        f"stack_ids: {', '.join(str(stack['stack_id']) for stack in stacks)}",
        f"dry_run: {REPRODUCTION_DRY_RUN}",
        f"output_directory: {RUN_DIR}",
    ]) + "\n",
    encoding="utf-8",
)
print("Saved final run manifest:", RUN_DIR / "run_manifest.json")
'''
        ),
    ]
    nbf.write(nb, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
