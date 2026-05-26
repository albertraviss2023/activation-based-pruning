# Custom Pruning Methods

ReduCNN is intentionally open-ended: bundled methods are examples and defaults,
not limits. Researchers can register any scoring math that returns one score per
output channel/filter.

## Core Contract

A pruning method receives layer/model context and returns a one-dimensional
array-like object:

```python
scores.shape == (number_of_output_channels,)
```

Higher scores are treated as more important. During mask building, ReduCNN keeps
the highest-scoring channels and prunes the lowest-scoring channels.

## Minimal Example

```python
import numpy as np
from reducnn.pruner import register_method

@register_method("my_l2_rms", framework="global")
def my_l2_rms(layer, tools=None, **kwargs):
    return np.asarray(tools.weight_l2(layer, mode="rms")).reshape(-1)
```

Use it:

```python
from reducnn.pruner import ReduCNNPruner

pruner = ReduCNNPruner(method="my_l2_rms", scope="local")
pruned_model, masks, duration = pruner.prune(model, loader, ratio=0.3, adapter=adapter)
```

## Framework-Specific Methods

Use `framework="global"` when the method works through ReduCNN helper tools.
Use `framework="torch"` or `framework="keras"` when you need direct backend
APIs.

```python
@register_method("my_method", framework="torch")
def my_method_torch(layer, tools=None, **kwargs):
    weight = layer.weight.detach().cpu().numpy()
    return np.sum(np.abs(weight), axis=(1, 2, 3))

@register_method("my_method", framework="keras")
def my_method_keras(layer, tools=None, **kwargs):
    weight = layer.get_weights()[0]
    return np.sum(np.abs(weight), axis=(0, 1, 2))
```

## Method Context

Depending on the backend path, ReduCNN may pass:

- `layer`
- `layer_name`
- `model`
- `loader`
- `device`
- `tools`
- `prunables`
- config values such as `calib_batches`, `chip_max_spatial`, or your own keys

Functions can accept only the arguments they need. The registry filters
arguments safely unless your function accepts `**kwargs`.

## Useful `tools` Helpers

`tools.collect_layer_outputs(layer, include_labels=True)`:
Collects calibration activations for a layer.

`tools.channel_matrix(act)`:
Converts activations into a channel-by-feature matrix.

`tools.pooled_nc(act)`:
Returns pooled activations shaped like samples by channels.

`tools.weight_l2(layer, mode="sum" | "rms")`:
Computes backend-aware per-channel weight energy.

`tools.chip_scores(act)`:
Computes CHIP-style channel independence scores.

`tools.rank_scores(act)`:
Computes HRank-style activation rank scores.

`tools.spectral_energy_scores(act)`:
Computes frequency-domain activation energy.

`tools.classwise_taylor_matrix(layer)`:
Computes class-wise Taylor contribution estimates when labels are available.

## UI Registration

ReduCNN Studio loads custom methods from `custom_methods/`.

Create a file such as `custom_methods/my_methods.py`:

```python
from reducnn.pruner import register_method

METHOD_METADATA = {
    "my_l2_rms": {
        "label": "My L2 RMS",
        "description": "Ranks channels by RMS weight energy.",
    }
}

@register_method("my_l2_rms", framework="global")
def my_l2_rms(layer, tools=None, **kwargs):
    return tools.weight_l2(layer, mode="rms")
```

Then run:

```bash
docker compose up --build reducnn-ui
```

The UI imports non-underscore `.py` files from `custom_methods/` and shows
registered methods in the pruning-method dropdown.

## Debugging

List methods:

```python
from reducnn.pruner import list_methods, list_method_names

print(list_methods("torch"))
print(list_method_names("keras"))
```

If a method is not found:

1. Make sure the Python file or notebook cell defining the method ran.
2. Check the framework name: `global`, `torch`, or `keras`.
3. Check the method name passed to `ReduCNNPruner(method=...)`.
4. Return one score per output channel.
5. Avoid returning `None` unless the layer should be skipped.
