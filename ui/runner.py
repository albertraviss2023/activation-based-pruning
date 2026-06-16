"""Execution helpers for the ReduCNN Streamlit UI."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Dict

from pruning_methods import METHOD_CATALOG, get_method_catalog, register_ui_methods


DATASETS = {
    "cifar10": {
        "label": "CIFAR-10",
        "num_classes": 10,
        "input_shape": (3, 32, 32),
        "source": "torchvision",
    },
    "cifar100": {
        "label": "CIFAR-100",
        "num_classes": 100,
        "input_shape": (3, 32, 32),
        "source": "torchvision",
    },
    "cats_dogs": {
        "label": "Cat vs Dog",
        "num_classes": 2,
        "input_shape": (3, 224, 224),
        "source": "imagefolder",
    },
}

MODELS = {
    "resnet18": "ResNet-18",
    "vgg16": "VGG-16",
    "densenet121": "DenseNet-121",
    "mobilenet_v2": "MobileNetV2",
}


@dataclass
class PruningJobConfig:
    backend: str
    dataset: str
    model: str
    method: str
    scope: str
    ratio: float
    epochs: int
    finetune_epochs: int
    batch_size: int
    calibration_batches: int
    learning_rate: float
    pretrained: bool
    baseline_mode: str
    baseline_checkpoint_path: str
    save_baseline: bool
    save_raw_pruned: bool
    save_finetuned: bool
    save_plots: bool
    cats_dogs_dir: str
    custom_methods_path: str
    smoke_mode: bool
    output_dir: str


def load_custom_method_modules(custom_methods_path: str = "custom_methods") -> Dict[str, str]:
    """Imports user method modules so their @register_method decorators run."""
    loaded: Dict[str, str] = {}
    if not custom_methods_path:
        return loaded

    for raw_path in str(custom_methods_path).split(";"):
        path = Path(raw_path.strip())
        if not path.exists():
            continue
        files = [path] if path.is_file() and path.suffix == ".py" else sorted(path.glob("*.py"))
        for file_path in files:
            if file_path.name.startswith("_"):
                continue
            module_name = f"reducnn_ui_custom_{file_path.stem}_{abs(hash(str(file_path.resolve())))}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    loaded[str(file_path)] = "Unable to create import spec"
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                metadata = getattr(module, "METHOD_METADATA", None)
                if isinstance(metadata, dict):
                    for method_name, method_meta in metadata.items():
                        if isinstance(method_meta, dict):
                            METHOD_CATALOG[str(method_name)] = {
                                "label": str(method_meta.get("label", method_name)),
                                "description": str(method_meta.get("description", "Registered custom pruning method.")),
                            }
                loaded[str(file_path)] = "loaded"
            except Exception as exc:
                loaded[str(file_path)] = f"failed: {exc}"
    return loaded


def ensure_methods_registered(custom_methods_path: str = "custom_methods") -> Dict[str, str]:
    register_ui_methods()
    return load_custom_method_modules(custom_methods_path)


def available_methods(framework: str = "torch", custom_methods_path: str = "custom_methods") -> Dict[str, Dict[str, str]]:
    framework = {"pytorch": "torch", "pt": "torch", "tensorflow": "keras", "tf": "keras"}.get(framework, framework)
    ensure_methods_registered(custom_methods_path)
    return get_method_catalog(framework)


def runtime_status() -> Dict[str, Any]:
    """Return lightweight runtime information for local, Docker, and Colab UI runs."""
    colab_runtime = bool(os.environ.get("COLAB_RELEASE_TAG")) or Path("/content").exists()
    status: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch_available": False,
        "torch_version": "not installed",
        "cuda_available": False,
        "device": "CPU",
        "device_count": 0,
        "colab_runtime": colab_runtime,
    }
    try:
        import torch

        status["torch_available"] = True
        status["torch_version"] = str(torch.__version__)
        status["cuda_available"] = bool(torch.cuda.is_available())
        if status["cuda_available"]:
            status["device_count"] = int(torch.cuda.device_count())
            status["device"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        status["error"] = str(exc)
    return status


def default_output_dir() -> str:
    """Choose an artifact directory that works well for local, Docker, and Colab runs."""
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        return str(drive_root / "reducnn" / "ui_runs")
    if bool(os.environ.get("COLAB_RELEASE_TAG")) or Path("/content").exists():
        return "/content/reducnn/ui_runs"
    return "outputs/ui_runs"


def load_torch_data(config: PruningJobConfig):
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    ds = DATASETS[config.dataset]
    image_size = ds["input_shape"][1]
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    if config.smoke_mode:
        n = max(config.batch_size * max(config.calibration_batches, 1), config.batch_size)
        x = torch.randn(n, *ds["input_shape"])
        y = torch.randint(0, ds["num_classes"], (n,))
        tensor_ds = torch.utils.data.TensorDataset(x, y)
        loader = DataLoader(tensor_ds, batch_size=config.batch_size)
        return loader, loader

    root = Path("data")
    if config.dataset == "cifar10":
        train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        val = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    elif config.dataset == "cifar100":
        train = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
        val = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    else:
        data_dir = Path(config.cats_dogs_dir or "data/cats_dogs/train")
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Cat vs Dog folder not found at {data_dir}. Expected ImageFolder layout with class subfolders."
            )
        full = datasets.ImageFolder(root=data_dir, transform=transform)
        val_len = max(1, int(0.15 * len(full)))
        train_len = max(1, len(full) - val_len)
        train, val = torch.utils.data.random_split(
            full,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42),
        )

    if config.calibration_batches > 0:
        cap = max(config.batch_size * config.calibration_batches, config.batch_size)
        if len(train) > cap:
            train = Subset(train, range(cap))
        if len(val) > cap:
            val = Subset(val, range(cap))

    return (
        DataLoader(train, batch_size=config.batch_size, shuffle=True, num_workers=2),
        DataLoader(val, batch_size=config.batch_size, shuffle=False, num_workers=2),
    )


def _save_layer_sensitivity_artifacts(masks: Dict[str, Any], output_dir: Path, run_id: str, model: str, method: str) -> Dict[str, str]:
    import csv
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not masks:
        return {}

    layers = sorted(masks.keys())
    rows = []
    for layer in layers:
        mask = np.asarray(masks[layer]).astype(bool).reshape(-1)
        kept = int(mask.sum())
        total = int(mask.size)
        rows.append(
            {
                "layer": layer,
                "kept": kept,
                "total": total,
                "keep_ratio": kept / max(total, 1),
                "pruned_ratio": 1.0 - (kept / max(total, 1)),
            }
        )

    csv_path = output_dir / f"{run_id}_{model}_{method}_layer_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["layer", "kept", "total", "keep_ratio", "pruned_ratio"])
        writer.writeheader()
        writer.writerows(rows)

    keep_pct = [100.0 * row["keep_ratio"] for row in rows]
    n_layers = len(rows)
    fig_w = float(np.clip(0.24 * n_layers + 8.0, 12.0, 34.0))
    fig_h = 6.5 if n_layers > 80 else 5.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x = np.arange(n_layers)
    colors = plt.cm.RdYlGn(np.asarray(keep_pct) / 100.0)
    ax.bar(x, keep_pct, color=colors, alpha=0.88, edgecolor="black", linewidth=0.4)
    avg_keep = float(np.mean(keep_pct))
    ax.axhline(avg_keep, color="#2563eb", linestyle="--", linewidth=1.3, label=f"Avg keep: {avg_keep:.1f}%")
    ax.set_title(f"{model} / {method} layer sensitivity", fontweight="bold")
    ax.set_ylabel("Filters kept (%)")
    if n_layers <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels([row["layer"] for row in rows], rotation=45, ha="right", fontsize=8)
    else:
        step = max(1, int(np.ceil(n_layers / 28)))
        ticks = x[::step]
        ax.set_xticks(ticks)
        ax.set_xticklabels([rows[int(i)]["layer"] for i in ticks], rotation=60, ha="right", fontsize=8)
        ax.set_xlabel(f"Layer index, showing every {step}th label")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend()
    fig.tight_layout()

    plot_path = output_dir / f"{run_id}_{model}_{method}_layer_sensitivity.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"layer_sensitivity_csv": str(csv_path), "layer_sensitivity_plot": str(plot_path)}


def _save_metrics_plot(summary: Dict[str, Any], output_dir: Path, run_id: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["Params (M)", "FLOPs (M)", "Accuracy (%)"]
    baseline = [
        summary["baseline_params"] / 1e6,
        summary["baseline_flops"] / 1e6,
        summary.get("baseline_accuracy_pct", 0.0),
    ]
    pruned = [
        summary["pruned_params"] / 1e6,
        summary["pruned_flops"] / 1e6,
        summary["final_accuracy_pct"],
    ]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, baseline, width, label="Baseline", color="#64748b")
    ax.bar(x + width / 2, pruned, width, label="Pruned", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend()
    ax.set_title("Baseline vs pruned model", fontweight="bold")
    fig.tight_layout()
    plot_path = output_dir / f"{run_id}_metrics_comparison.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(plot_path)


def run_torch_job(config: PruningJobConfig) -> Dict[str, Any]:
    from reducnn.backends.torch_backend import PyTorchAdapter
    from reducnn.pruner import ReduCNNPruner

    ensure_methods_registered(config.custom_methods_path)
    ds = DATASETS[config.dataset]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")

    train_loader, val_loader = load_torch_data(config)
    adapter_config = {
        "backend": "pytorch",
        "dataset": config.dataset,
        "dataset_key": config.dataset,
        "model_type": config.model,
        "input_shape": ds["input_shape"],
        "num_classes": ds["num_classes"],
        "ratio": config.ratio,
        "method": config.method,
        "scope": config.scope,
        "lr": config.learning_rate,
        "epochs": config.epochs,
        "ft_epochs": config.finetune_epochs,
        "prune_batches": config.calibration_batches,
        "baseline_checkpoint_policy": "off",
    }
    adapter = PyTorchAdapter(adapter_config)
    model = adapter.get_model(
        config.model,
        input_shape=ds["input_shape"],
        num_classes=ds["num_classes"],
        pretrained=config.pretrained,
    )

    baseline_path = Path(config.baseline_checkpoint_path) if config.baseline_checkpoint_path else None
    baseline_mode = str(config.baseline_mode).lower().strip()
    baseline_artifact = None
    baseline_history = None

    if baseline_mode == "load_checkpoint":
        if baseline_path is None or not baseline_path.exists():
            raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_path}")
        adapter.load_checkpoint(model, str(baseline_path))
        baseline_artifact = str(baseline_path)
    elif baseline_mode == "auto_latest":
        latest = adapter._latest_baseline_ckpt(model)
        if latest is not None and latest.exists():
            adapter.load_checkpoint(model, str(latest))
            baseline_artifact = str(latest)
        elif config.epochs > 0:
            baseline_history = adapter.train(model, train_loader, config.epochs, "Baseline", val_loader=val_loader, plot=False)
    elif baseline_mode == "train_new" and config.epochs > 0:
        baseline_history = adapter.train(model, train_loader, config.epochs, "Baseline", val_loader=val_loader, plot=False)

    if config.save_baseline:
        baseline_out = output_dir / f"{run_id}_{config.model}_{config.dataset}_baseline.pth"
        adapter.save_checkpoint(model, str(baseline_out))
        baseline_artifact = str(baseline_out)

    before = adapter.get_stats(model, val_loader)
    baseline_acc = adapter.evaluate(model, val_loader)

    raw_pruned_path = output_dir / f"{run_id}_{config.model}_{config.method}_pruned_raw.pth"
    final_path = output_dir / f"{run_id}_{config.model}_{config.method}_finetuned.pth"

    pruner = ReduCNNPruner(method=config.method, scope=config.scope, config=adapter_config)
    pruned_model, masks, prune_duration = pruner.prune(
        model,
        train_loader,
        ratio=config.ratio,
        adapter=adapter,
        save_pruned_path=str(raw_pruned_path) if config.save_raw_pruned else None,
    )

    finetune_history = None
    if config.finetune_epochs > 0:
        finetune_history = adapter.train(
            pruned_model,
            train_loader,
            config.finetune_epochs,
            "Pruned",
            val_loader=val_loader,
            plot=False,
        )

    if config.save_finetuned:
        adapter.save_checkpoint(pruned_model, str(final_path))

    after = adapter.get_stats(pruned_model, val_loader)
    final_acc = adapter.evaluate(pruned_model, val_loader)

    summary = {
        "run_id": run_id,
        "backend": "pytorch",
        "dataset": config.dataset,
        "model": config.model,
        "method": config.method,
        "scope": config.scope,
        "ratio": config.ratio,
        "baseline_flops": before[0],
        "baseline_params": before[1],
        "baseline_accuracy_pct": baseline_acc,
        "pruned_flops": after[0],
        "pruned_params": after[1],
        "param_reduction_pct": 100.0 * (1.0 - (after[1] / max(before[1], 1.0))),
        "flop_reduction_pct": 100.0 * (1.0 - (after[0] / max(before[0], 1.0))) if before[0] else 0.0,
        "final_accuracy_pct": final_acc,
        "accuracy_delta_pct": final_acc - baseline_acc,
        "prune_duration_sec": prune_duration,
        "layers_pruned": len(masks),
        "artifacts": {
            "baseline_checkpoint": baseline_artifact,
            "pruned_checkpoint": str(raw_pruned_path) if config.save_raw_pruned else None,
            "final_checkpoint": str(final_path) if config.save_finetuned else None,
        },
    }
    if baseline_history is not None:
        summary["baseline_history"] = baseline_history
    if finetune_history is not None:
        summary["finetune_history"] = finetune_history

    if config.save_plots:
        summary["artifacts"].update(_save_layer_sensitivity_artifacts(masks, output_dir, run_id, config.model, config.method))
        summary["artifacts"]["metrics_plot"] = _save_metrics_plot(summary, output_dir, run_id)

    summary_path = output_dir / f"{run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    summary["artifacts"]["summary_json"] = str(summary_path)
    return summary


def estimate_pruning_plan(config: PruningJobConfig) -> Dict[str, Any]:
    ds = DATASETS[config.dataset]
    method = available_methods(config.backend, config.custom_methods_path).get(
        config.method,
        {"label": config.method, "description": "Registered custom pruning method."},
    )
    return {
        "Dataset": ds["label"],
        "Model": MODELS[config.model],
        "Method": method["label"],
        "Scope": config.scope,
        "Target pruning": f"{config.ratio:.0%}",
        "Baseline source": config.baseline_mode,
        "Input shape": "x".join(str(v) for v in ds["input_shape"]),
        "Classes": ds["num_classes"],
        "Calibration batches": config.calibration_batches,
        "Output folder": config.output_dir,
    }


def export_job_config(config: PruningJobConfig) -> str:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    payload = {
        "run_id": run_id,
        "backend": config.backend,
        "dataset": config.dataset,
        "model": config.model,
        "method": config.method,
        "scope": config.scope,
        "ratio": config.ratio,
        "epochs": config.epochs,
        "finetune_epochs": config.finetune_epochs,
        "batch_size": config.batch_size,
        "calibration_batches": config.calibration_batches,
        "learning_rate": config.learning_rate,
        "pretrained": config.pretrained,
        "baseline_mode": config.baseline_mode,
        "baseline_checkpoint_path": config.baseline_checkpoint_path,
        "save_baseline": config.save_baseline,
        "save_raw_pruned": config.save_raw_pruned,
        "save_finetuned": config.save_finetuned,
        "save_plots": config.save_plots,
        "cats_dogs_dir": config.cats_dogs_dir,
        "custom_methods_path": config.custom_methods_path,
        "smoke_mode": config.smoke_mode,
        "output_dir": config.output_dir,
    }
    path = output_dir / f"{run_id}_colab_job_config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)
