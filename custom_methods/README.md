# Custom ReduCNN Methods

Drop Python files in this folder to make methods available in ReduCNN Studio.
The UI imports every non-underscore `.py` file in this directory, so each file
can register one or more methods with the normal package decorator.

Example:

```python
from reducnn.pruner import register_method

@register_method("my_method", framework="global")
def my_method(layer, tools=None, **kwargs):
    return tools.weight_l2(layer)
```

Registered method functions can receive any context exposed by the backend:
`layer`, `layer_name`, `model`, `loader`, `device`, `tools`, `prunables`, and
config values. Higher scores mean the channel is more important and should be
kept.
