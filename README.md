# ReduCNN

ReduCNN is a dual-framework research package for activation-based structural
pruning of convolutional neural networks. It supports PyTorch and Keras models,
lets researchers register their own pruning mathematics, and performs physical
channel/filter removal so the pruned model is actually smaller.

The core idea is simple:

1. Build or load a CNN.
2. Score each channel/filter with a pruning method.
3. Build keep masks from the scores.
4. Apply structural surgery.
5. Fine-tune and compare accuracy, parameters, and FLOPs.

## Install

For local research:

```bash
git clone https://github.com/albertraviss2023/activation-based-pruning.git
cd activation-based-pruning
pip install -e .
```

Install the framework you want to use:

```bash
pip install -e ".[torch]"
```

or:

```bash
pip install -e ".[keras]"
```

For development tools:

```bash
pip install -e ".[dev]"
```

For the Dockerized UI dependencies outside Docker:

```bash
pip install -e ".[ui]"
```

For Colab GPU use, install the same UI extras inside the Colab runtime:

```python
!pip install -q -e ".[ui]"
```

## Quick Start: PyTorch

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from reducnn.backends.torch_backend import PyTorchAdapter
from reducnn.pruner import ReduCNNPruner

transform = transforms.Compose([transforms.ToTensor()])
train = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
loader = DataLoader(train, batch_size=32, shuffle=True)

config = {
    "backend": "pytorch",
    "dataset": "cifar10",
    "model_type": "resnet18",
    "input_shape": (3, 32, 32),
    "num_classes": 10,
    "prune_batches": 8,
}

adapter = PyTorchAdapter(config)
model = adapter.get_model("resnet18", input_shape=(3, 32, 32), num_classes=10)

pruner = ReduCNNPruner(method="apoz", scope="local", config=config)
pruned_model, masks, duration = pruner.prune(model, loader, ratio=0.3, adapter=adapter)

before = adapter.get_stats(model, loader)
after = adapter.get_stats(pruned_model, loader)

print("Original params:", before[1])
print("Pruned params:", after[1])
```

## Quick Start: Keras

```python
import tensorflow as tf

from reducnn.backends.keras_backend import KerasAdapter
from reducnn.pruner import ReduCNNPruner

(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
y_train = y_train.reshape(-1)
loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

config = {
    "backend": "keras",
    "dataset": "cifar10",
    "model_type": "vgg16",
    "input_shape": (32, 32, 3),
    "num_classes": 10,
    "prune_batches": 8,
}

adapter = KerasAdapter(config)
model = adapter.get_model("vgg16", input_shape=(32, 32, 3), num_classes=10)

pruner = ReduCNNPruner(method="l1_norm", scope="local", config=config)
pruned_model, masks, duration = pruner.prune(model, loader, ratio=0.2, adapter=adapter)
```

## Register Custom Pruning Math

ReduCNN is designed so researchers are not limited to bundled methods. Any
function can become a pruning method if it returns a one-dimensional score array
for a layer. Higher scores mean "more important, keep this channel."

```python
import numpy as np
from reducnn.pruner import register_method

@register_method("my_activation_energy", framework="global")
def my_activation_energy(layer, tools=None, **kwargs):
    act, _ = tools.collect_layer_outputs(layer, include_labels=False)
    if act is None:
        return tools.weight_l2(layer)
    channel_matrix = tools.channel_matrix(act)
    return np.mean(np.square(channel_matrix), axis=1)
```

Then use it like any built-in method:

```python
pruner = ReduCNNPruner(method="my_activation_energy", scope="global")
pruned_model, masks, duration = pruner.prune(model, loader, ratio=0.3, adapter=adapter)
```

You can inspect what is registered:

```python
from reducnn.pruner import list_method_names

print(list_method_names("torch"))
```

Framework values:
- `global`: method can be used by both backends.
- `torch`: PyTorch-specific method.
- `keras`: Keras-specific method.

Useful arguments that may be passed to custom methods:
- `layer`
- `layer_name`
- `model`
- `loader`
- `device`
- `tools`
- `prunables`
- any values from your config dictionary

The `tools` object provides helpers for activation collection, channel matrix
conversion, weight norms, CHIP-style scores, class-wise Taylor matrices, and
other reusable scoring utilities.

## Dockerized UI

ReduCNN Studio is a Streamlit app for configuring pruning runs through a clean
interface. It supports:

- model selection
- dataset selection: Cat vs Dog, CIFAR-10, CIFAR-100
- pruning method selection from the live ReduCNN registry
- custom method loading from `custom_methods/`
- baseline selection: load latest, train new, load checkpoint, or use model initialization
- checkpoint saving for baseline, raw pruned, and fine-tuned models
- layer sensitivity plots and CSV tables saved to disk
- smoke-mode synthetic runs
- checkpoint and summary artifact creation

The app reports the active runtime at the top of the page. For GPU pruning, it
should show `Runtime: CUDA GPU`.

Start it with:

```bash
docker compose up --build reducnn-ui
```

Open:

```text
http://localhost:8501
```

UI artifacts are written to `outputs/ui_runs/` by default. The Docker Compose
configuration mounts `data/`, `outputs/`, `saved_models/`, and `custom_methods/`
so files created in the container are visible in the repo workspace.

Custom methods for the UI:

1. Add a `.py` file under `custom_methods/`.
2. Register methods with `@register_method(...)`.
3. Optionally add `METHOD_METADATA` for a nicer UI label.
4. Restart or refresh the app.

Example:

```python
from reducnn.pruner import register_method

METHOD_METADATA = {
    "my_energy_score": {
        "label": "My Energy Score",
        "description": "Ranks channels by RMS weight energy.",
    }
}

@register_method("my_energy_score", framework="global")
def my_energy_score(layer, tools=None, **kwargs):
    return tools.weight_l2(layer, mode="rms")
```

### Running the UI on Colab GPU

To use a Colab GPU from the UI, run Streamlit inside the Colab runtime and open
it through a tunnel. A local Docker container cannot borrow the Colab GPU.

Quick Colab shape:

```python
!git clone https://github.com/albertraviss2023/activation-based-pruning.git
%cd activation-based-pruning
!pip install -q -e ".[ui]"
!streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true > /content/reducnn_streamlit.log 2>&1 &
```

Then expose it:

```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cloudflared
!chmod +x /content/cloudflared
!/content/cloudflared tunnel --url http://localhost:8501
```

Open the printed `trycloudflare.com` URL and confirm that the UI says
`Runtime: CUDA GPU`. Full steps are in
[Running ReduCNN Studio on Google Colab](docs/COLAB_UI.md).

## High-Level Orchestration

Use `Orchestrator` when you want a train, prune, fine-tune flow:

```python
from reducnn.engine import Orchestrator

config = {
    "backend": "pytorch",
    "dataset": "cifar10",
    "model_type": "resnet18",
    "input_shape": (3, 32, 32),
    "num_classes": 10,
    "method": "apoz",
    "scope": "local",
    "ratio": 0.3,
    "epochs": 5,
    "ft_epochs": 3,
}

orchestrator = Orchestrator(config)
pruned_model, masks = orchestrator.run(train_loader, val_loader=val_loader)
```

## Outputs

Common output locations:

- `saved_models/baselines/<backend>/<dataset>/<model>/`
- `saved_models/pruned_raw/<backend>/<dataset>/<model>/<method>/`
- `saved_models/fine_tuned/<backend>/<dataset>/<model>/<method>/`
- `outputs/experiments/<dataset>/<model>/<run_id>/`
- `outputs/ui_runs/`

These directories are ignored by git by default.

## Repo Layout

```text
src/reducnn/
  analyzer/        method comparison, validation, Pareto analysis
  backends/        PyTorch and Keras adapters
  core/            shared adapter/storage/decorator utilities
  engine/          high-level orchestration
  pruner/          registry, scoring, masks, structural surgery
  visualization/   reporting and diagnostic visualizations

custom_methods/    drop-in methods loaded by the UI
ui/                Dockerized Streamlit app
docs/              workflow and project documentation
examples/          script-based examples
tests/             regression and workflow tests
```

## More Documentation

- [Documentation Index](docs/README.md)
- [Workflow How-To](docs/WORKFLOWS_HOWTO.md)
- [Custom Methods](docs/CUSTOM_METHODS.md)
- [Method Math Notes](docs/METHOD_MATH.md)
- [Adaptive Hybrid Method](docs/ADAPTIVE_HYBRID_METHOD.md)
- [Objective LFPC Experiments](docs/OBJECTIVE_LFPC_EXPERIMENTS.md)
- [Experiment Metadata Registry](docs/EXPERIMENT_METADATA_REGISTRY.md)
- [Experiment Metrics Schema](docs/EXPERIMENT_METRICS_SCHEMA.md)
- [UI and GPU Execution](docs/UI_GPU_GUIDE.md)
- [Running ReduCNN Studio on Google Colab](docs/COLAB_UI.md)
- [Repo Hygiene](docs/REPO_HYGIENE.md)
- [Module Documentation](MODULE_DOCUMENTATION.md)
- [Implementation Audit](docs/IMPLEMENTATION_AUDIT_v0.6.6.md)
- [Literature Fidelity Report](docs/LITERATURE_FIDELITY_REPORT_v2.md)

## Development Checks

```bash
python -m compileall src ui custom_methods
pytest
```

For a quick UI wiring check, use ReduCNN Studio with `Smoke mode` enabled.
