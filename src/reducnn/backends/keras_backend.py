import numpy as np
from typing import Dict, Any, Callable, Tuple, Optional, List
import os
import copy
from pathlib import Path
from datetime import datetime

from ..core.adapter import FrameworkAdapter
from ..core.decorators import timer
from ..core.exceptions import SurgeryError
from ..pruner.registry import get_method, call_score_fn
from ..pruner.custom_method_tools import CustomMethodTools

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models as k_models
except ImportError:
    # Framework-specific imports are wrapped in try-except to maintain
    # the library's ability to run (or be documented) without all backends.
    pass

class KerasStructuralPruner:
    """Specialized surgeon for Keras functional models with branching support."""
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.dependencies = self._trace(model)

    def _trace(self, model: tf.keras.Model) -> Dict[str, Any]:
        """Traces dependencies to identify residual clusters in Keras."""
        nodes_info = {}
        conv_names = set()

        def producer_layer(tensor):
            kh = tensor._keras_history
            return getattr(kh, "operation", getattr(kh, "layer", None))

        def nearest_upstream_convs(inbound_tensor) -> List[str]:
            queue = list(inbound_tensor) if isinstance(inbound_tensor, list) else [inbound_tensor]
            visited_tensors = set()
            found = set()
            while queue:
                t = queue.pop(0)
                if id(t) in visited_tensors:
                    continue
                visited_tensors.add(id(t))
                layer_obj = producer_layer(t)
                if isinstance(layer_obj, layers.Conv2D):
                    found.add(layer_obj.name)
                    continue
                if hasattr(layer_obj, "input"):
                    in_t = layer_obj.input
                    if isinstance(in_t, list):
                        queue.extend(in_t)
                    else:
                        queue.append(in_t)
            return sorted(found)
        
        # 1. Identify prunable nodes and their structural metadata
        for layer in model.layers:
            if isinstance(layer, layers.Conv2D):
                nodes_info[layer.name] = {
                    "type": "conv2d",
                    "inputs": [],
                    "outputs": [],
                    "cluster": None
                }
                conv_names.add(layer.name)

        # 1.5 Populate conv-to-conv inputs/outputs through non-prunable layers.
        for layer in model.layers:
            if isinstance(layer, layers.Conv2D):
                upstream = [n for n in nearest_upstream_convs(layer.input) if n in conv_names and n != layer.name]
                nodes_info[layer.name]["inputs"] = upstream

        for name in conv_names:
            nodes_info[name]["outputs"] = []
        for child_name, info in nodes_info.items():
            for parent_name in info.get("inputs", []):
                if parent_name in nodes_info and child_name not in nodes_info[parent_name]["outputs"]:
                    nodes_info[parent_name]["outputs"].append(child_name)
        
        # 2. Identify clusters by looking for Add layers (Residual)
        # Concatenate layers in DenseNet do NOT require identical masks.
        clusters = {}
        cluster_id = 0
        raw_clusters = []
        for layer in model.layers:
            if isinstance(layer, layers.Add):
                cluster_members = []
                queue = list(layer.input) if isinstance(layer.input, list) else [layer.input]
                visited = set()
                
                while queue:
                    t = queue.pop(0)
                    if id(t) in visited: continue
                    visited.add(id(t))
                    
                    prod_layer = getattr(t._keras_history, "operation", getattr(t._keras_history, "layer", None))
                    if isinstance(prod_layer, layers.Conv2D):
                        cluster_members.append(prod_layer.name)
                    elif hasattr(prod_layer, "input"):
                        in_t = prod_layer.input
                        if isinstance(in_t, list): queue.extend(in_t)
                        else: queue.append(in_t)
                
                if len(cluster_members) > 1:
                    raw_clusters.append(set(cluster_members))
        
        # Merge overlapping clusters
        merged = []
        for c in raw_clusters:
            found = False
            for m in merged:
                if not c.isdisjoint(m):
                    m.update(c)
                    found = True
                    break
            if not found: merged.append(c)
            
        for idx, cluster in enumerate(merged):
            for name in cluster:
                if name in nodes_info:
                    nodes_info[name]["cluster"] = idx
            clusters[idx] = list(cluster)
            
        return {"nodes": nodes_info, "clusters": clusters}

    def harmonize_masks(self, masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Ensures layers in residual clusters share masks only when channel sizes match."""
        h_masks = copy.deepcopy(masks)
        for c_id, members in self.dependencies["clusters"].items():
            # Only harmonize layers that have the same number of output channels.
            # Residual subgraphs may include upstream convs with different widths.
            size_groups: Dict[int, List[str]] = {}
            for m in members:
                layer_obj = self.model.get_layer(m)
                ch = int(getattr(layer_obj, "filters", 0))
                if ch <= 0:
                    continue
                size_groups.setdefault(ch, []).append(m)

            for _ch, same_size_members in size_groups.items():
                anchor_name = next((m for m in same_size_members if m in h_masks), None)
                if anchor_name is None:
                    continue
                anchor_mask = np.asarray(h_masks[anchor_name]).astype(bool).reshape(-1)
                for m in same_size_members:
                    h_masks[m] = anchor_mask.copy()
        return h_masks

class KerasLoaderWrapper:
    """Wraps a data loader (e.g., PyTorch DataLoader) to ensure it returns 
    NumPy arrays and handles channel transposition if necessary.
    """
    def __init__(self, loader, target_shape: Tuple[int, int, int] = None):
        self.loader = loader
        self.target_shape = target_shape

    def __iter__(self):
        for batch in self.loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch, None
            
            # Convert PyTorch/TF tensors to NumPy
            if hasattr(x, 'detach'): x = x.detach().cpu().numpy()
            elif hasattr(x, 'numpy'): x = x.numpy()
            
            if y is not None:
                if hasattr(y, 'detach'): y = y.detach().cpu().numpy()
                elif hasattr(y, 'numpy'): y = y.numpy()
            
            # Check if we need to transpose (C, H, W) -> (H, W, C)
            if self.target_shape and len(x.shape) == 4:
                # If target is (H, W, C) but x is (N, C, H, W)
                if x.shape[1] == self.target_shape[2] and x.shape[2] == self.target_shape[0] and x.shape[3] == self.target_shape[1]:
                    x = x.transpose(0, 2, 3, 1)
                # If target is (C, H, W) but x is (N, H, W, C)
                elif x.shape[3] == self.target_shape[0] and x.shape[1] == self.target_shape[1] and x.shape[2] == self.target_shape[2]:
                    x = x.transpose(0, 3, 1, 2)
            
            if y is not None:
                yield x, y
            else:
                yield x

    def __len__(self):
        return len(self.loader)

class KerasAdapter(FrameworkAdapter):
    """Bridge between the core pruner and the Keras/TensorFlow framework."""
    def __init__(self, config: dict = None):
        self.config = config or {}

    def _resolve_prune_batches(self, loader: Any) -> int:
        """Resolves how many calibration batches to use for scoring.

        Policy:
        - If user explicitly sets a batch limit, honor it.
        - Otherwise, default to a full pass over the calibration loader.
        - If loader length is unknown (e.g., generator), use a safe fallback cap.
        """
        cfg = self.config or {}
        for key in ("prune_batches", "calib_batches", "calibration_batches"):
            val = cfg.get(key, None)
            if val is None or val == "":
                continue
            try:
                iv = int(val)
                if iv > 0:
                    return iv
            except Exception:
                pass

        try:
            n = int(len(loader))
            if n > 0:
                return n
        except Exception:
            pass

        return max(1, int(cfg.get("prune_batches_fallback", 128)))

    def _resolve_baseline_dir(self, model: Any) -> Path:
        cfg = self.config or {}
        model_type = str(
            cfg.get("model_type")
            or cfg.get("model_name")
            or getattr(model, "name", model.__class__.__name__.lower())
        ).lower().strip()
        dataset_key = str(
            cfg.get("dataset_key")
            or cfg.get("dataset")
            or os.environ.get("REDUCNN_DATASET_KEY")
            or "dataset"
        ).lower().strip()
        return Path("saved_models") / "baselines" / "keras" / dataset_key / model_type

    def _latest_baseline_ckpt(self, model: Any) -> Optional[Path]:
        d = self._resolve_baseline_dir(model)
        if not d.exists():
            return None
        files = sorted(d.glob("*.weights.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None

    def _new_baseline_ckpt(self, model: Any) -> Path:
        d = self._resolve_baseline_dir(model)
        d.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = d.name
        dataset_key = d.parent.name
        return d / f"{stamp}_keras_{model_type}_{dataset_key}_baseline.weights.h5"

    @staticmethod
    def _is_baseline_name(name: str) -> bool:
        n = str(name or "").lower()
        return ("baseline" in n) or ("base" in n and "heal" not in n and "finetune" not in n)

    def trace_graph(self, model: tf.keras.Model) -> Dict[str, Any]:
        return KerasStructuralPruner(model).dependencies

    def classify_architecture(self, model: tf.keras.Model) -> str:
        graph = self.trace_graph(model)
        if any(isinstance(l, layers.Add) for l in model.layers): return "residual"
        if any(isinstance(l, layers.Concatenate) for l in model.layers): return "concatenative"
        return "sequential"

    def _prepare_eval_loader(self, model: tf.keras.Model, loader: Any):
        """Builds Keras-compatible evaluate() input from generic loaders."""
        if loader is None:
            return None, {}
        if isinstance(loader, tf.data.Dataset):
            return loader, {}
        if isinstance(loader, tf.keras.utils.Sequence):
            return loader, {}
        wrapped = KerasLoaderWrapper(loader, model.input_shape[1:])
        kwargs = {}
        try:
            steps = int(len(wrapped))
            if steps > 0:
                kwargs["steps"] = steps
        except Exception:
            pass
        return iter(wrapped), kwargs

    def _prepare_fit_loader(self, model: tf.keras.Model, loader: Any, is_validation: bool = False):
        """Builds Keras-compatible fit() / validation_data input from generic loaders."""
        if loader is None:
            return None, {}
        if isinstance(loader, tf.data.Dataset):
            return loader, {}
        if isinstance(loader, tf.keras.utils.Sequence):
            return loader, {}
        wrapped = KerasLoaderWrapper(loader, model.input_shape[1:])
        kwargs = {}
        try:
            steps = int(len(wrapped))
            if steps > 0:
                if is_validation:
                    kwargs["validation_steps"] = steps
                else:
                    kwargs["steps_per_epoch"] = steps
        except Exception:
            pass
        return iter(wrapped), kwargs

    def _single_pass_multi_metric_scores(
        self, model: tf.keras.Model, loader: Any, metrics: List[str]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Collects bundled activation metrics in one calibration pass."""
        requested = {m.lower().strip() for m in metrics}
        supported = {"mean_abs_act", "apoz"}
        active = requested.intersection(supported)
        if not active:
            return {}

        conv_layers = [(l.name, l) for l in model.layers if isinstance(l, layers.Conv2D)]
        if not conv_layers:
            return {m: {} for m in active}

        wrapped_loader = KerasLoaderWrapper(loader, model.input_shape[1:])
        batches = self._resolve_prune_batches(wrapped_loader)
        model_input = model.inputs[0]
        out_tensors = [l.output for _, l in conv_layers]
        probe_model = k_models.Model(inputs=model_input, outputs=out_tensors)

        mean_abs_sum, mean_abs_cnt = {}, {}
        apoz_zero_sum, apoz_cnt = {}, {}

        for name, layer in conv_layers:
            c = int(layer.get_weights()[0].shape[-1])
            if "mean_abs_act" in active:
                mean_abs_sum[name] = np.zeros(c, dtype=np.float64)
                mean_abs_cnt[name] = 0
            if "apoz" in active:
                apoz_zero_sum[name] = np.zeros(c, dtype=np.float64)
                apoz_cnt[name] = 0

        it = iter(wrapped_loader)
        for _ in range(batches):
            try:
                batch = next(it)
            except StopIteration:
                break

            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            elif isinstance(batch, (list, tuple)):
                x, y = batch[0], None
            else:
                x, y = batch, None

            conv_outs = probe_model(x, training=False)
            if len(conv_layers) == 1:
                conv_outs = [conv_outs]

            for (name, _), a in zip(conv_layers, conv_outs):
                a_np = a.numpy()
                elems = int(a_np.shape[0] * a_np.shape[1] * a_np.shape[2])

                if "mean_abs_act" in active:
                    mean_abs_sum[name] += np.abs(a_np).sum(axis=(0, 1, 2))
                    mean_abs_cnt[name] += elems

                if "apoz" in active:
                    post_relu = np.maximum(a_np, 0.0)
                    apoz_zero_sum[name] += (post_relu == 0).sum(axis=(0, 1, 2))
                    apoz_cnt[name] += elems

        results: Dict[str, Dict[str, np.ndarray]] = {}
        if "mean_abs_act" in active:
            results["mean_abs_act"] = {
                n: (mean_abs_sum[n] / max(mean_abs_cnt[n], 1)).reshape(-1) for n, _ in conv_layers
            }
        if "apoz" in active:
            results["apoz"] = {
                n: (1.0 - (apoz_zero_sum[n] / max(apoz_cnt[n], 1))).reshape(-1) for n, _ in conv_layers
            }
        return results

    def get_multi_metric_scores(self, model: tf.keras.Model, loader: Any, metrics: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """Calculates multiple pruning metrics with shared data passes when possible."""
        metric_list = [str(m).lower().strip() for m in metrics]
        results: Dict[str, Dict[str, np.ndarray]] = {}

        data_dependent = {"mean_abs_act", "apoz"}
        requested_data = [m for m in metric_list if m in data_dependent]
        if requested_data:
            results.update(self._single_pass_multi_metric_scores(model, loader, requested_data))

        for m in metric_list:
            if m in results:
                continue
            try:
                results[m] = self.get_score_map(model, loader, m)
            except Exception:
                # Keep multi-metric visual diagnostics robust when optional methods
                # are not registered in a particular runtime.
                continue
        return results

    def get_global_activations(self, model: tf.keras.Model, loader: Any, num_batches: int = 1) -> Dict[str, np.ndarray]:
        """
        Collects average activation magnitudes for all prunable layers in a single pass.
        Useful for global network flow animations.
        """
        wrapped_loader = KerasLoaderWrapper(loader, model.input_shape[1:])
        conv_layers = [(l.name, l) for l in model.layers if isinstance(l, layers.Conv2D)]
        
        if not conv_layers:
            return {}
            
        out_tensors = []
        for n, l in conv_layers:
            out = l.output
            if len(out.shape) == 4:
                out = layers.GlobalAveragePooling2D()(out)
            out_tensors.append(out)
            
        act_model = tf.keras.Model(inputs=model.input, outputs=out_tensors)
        
        acts = {n: [] for n, _ in conv_layers}
        for i, batch in enumerate(wrapped_loader):
            if i >= num_batches: break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            preds = act_model.predict(x, verbose=0)
            if len(conv_layers) == 1: 
                preds = [preds]
            for j, (n, _) in enumerate(conv_layers):
                acts[n].append(preds[j])
                
        return {k: np.concatenate(v, axis=0).mean(axis=0) for k, v in acts.items()}

    def get_layer_activations(self, model: tf.keras.Model, loader: Any, layer_name: str, num_batches: int = 1) -> np.ndarray:
        """
        Collects raw activations for a specific layer.
        Returns array of shape [num_samples, channels]
        """
        wrapped_loader = KerasLoaderWrapper(loader, model.input_shape[1:])
        
        # Find the layer
        target_layer = model.get_layer(layer_name)
        # Create a sub-model that outputs both the layer output and global average pool
        # If it's a conv layer, we pool it
        out = target_layer.output
        if len(out.shape) == 4:
            # Use Keras layer for pooling to stay within the symbolic graph
            out = layers.GlobalAveragePooling2D()(out)
            
        activation_model = tf.keras.Model(inputs=model.input, outputs=out)
        
        activations = []
        for i, batch in enumerate(wrapped_loader):
            if i >= num_batches: break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            activations.append(activation_model.predict(x, verbose=0))
            
        return np.concatenate(activations, axis=0)

    def get_layer_importance(self, model: tf.keras.Model, loader: Any, layer_name: str, method: str = 'l1_norm') -> np.ndarray:
        """
        Computes importance scores for a specific layer using the requested method.
        """
        # Score calculation in Keras is usually internal to the adapter for efficiency.
        if method in ['l1_norm', 'l2_norm']:
            layer = model.get_layer(layer_name)
            weights = layer.get_weights()[0] # [k, k, in, out]
            if method == 'l1_norm':
                return np.sum(np.abs(weights), axis=(0, 1, 2))
            else:
                return np.sqrt(np.sum(np.square(weights), axis=(0, 1, 2)))
        
        # Bundled activation methods
        score_map = self._activation_scores(model, loader, method)
        if layer_name in score_map:
            return score_map[layer_name]
            
        raise ValueError(f"No importance scores found for layer {layer_name} using {method}.")

    def get_model(self, model_type: str, input_shape: Tuple[int, int, int] = None, 
                  num_classes: int = None, pretrained: bool = False, **kwargs) -> Any:
        original_model_type = str(model_type).lower().strip()
        model_type = original_model_type
        keras_lr = float(self.config.get('keras_lr', self.config.get('lr', 3e-4)))
        alias_map = {
            "resnet101": "resnet50",
            "resnet152": "resnet50",
            "densenet": "densenet121",
            "densenet169": "densenet121",
            "densenet201": "densenet121",
            "mobilenet_v2": "mobilenet",
        }
        model_type = alias_map.get(model_type, model_type)
        
        # Default input shape if none provided
        input_shape = input_shape or self.config.get('input_shape', (32, 32, 3))
        num_classes = num_classes or self.config.get('num_classes', 10)

        # Normalize input shape if it's in torch format (C, H, W) but we're in channels_last
        if input_shape[0] < input_shape[1] and input_shape[0] < input_shape[2]:
            if tf.keras.backend.image_data_format() == 'channels_last':
                # Convert (C, H, W) -> (H, W, C)
                input_shape = (input_shape[1], input_shape[2], input_shape[0])

        if model_type == 'vgg16':
            m = self._build_vgg16_bn_keras(input_shape=input_shape, num_classes=num_classes)
            m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=keras_lr),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return m

        if model_type in ('resnet18', 'resnet34'):
            if pretrained:
                print(
                    f"Warning: pretrained=True for '{model_type}' is not available in the "
                    "custom CIFAR-style Keras builder. Use a checkpoint for pretrained runs."
                )
            depth_cfg = (2, 2, 2, 2) if model_type == 'resnet18' else (3, 4, 6, 3)
            m = self._build_resnet_cifar_keras(
                input_shape=input_shape,
                num_classes=num_classes,
                blocks=depth_cfg,
                name=f"{model_type}_cifar",
            )
            m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=keras_lr),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return m

        if model_type in ('resnet50', 'resnet'):
            weights_cfg = str(self.config.get('keras_weights', 'none')).lower().strip()
            if weights_cfg in ('none', 'null', ''):
                weights = 'imagenet' if pretrained else None
            else:
                weights = weights_cfg
            base = tf.keras.applications.ResNet50(include_top=False, weights=weights, input_shape=input_shape)
            x = layers.GlobalAveragePooling2D()(base.output)
            x = layers.Dense(512, activation='relu')(x)
            out = layers.Dense(num_classes, activation='softmax')(x)
            m = k_models.Model(inputs=base.input, outputs=out)
            m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=keras_lr),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return m

        if model_type in ('densenet121', 'densenet'):
            weights_cfg = str(self.config.get('keras_weights', 'none')).lower().strip()
            if weights_cfg in ('none', 'null', ''):
                weights = 'imagenet' if pretrained else None
            else:
                weights = weights_cfg
            base = tf.keras.applications.DenseNet121(include_top=False, weights=weights, input_shape=input_shape)
            x = layers.GlobalAveragePooling2D()(base.output)
            x = layers.Dense(512, activation='relu')(x)
            out = layers.Dense(num_classes, activation='softmax')(x)
            m = k_models.Model(inputs=base.input, outputs=out)
            m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=keras_lr),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return m

        if model_type in ('mobilenet', 'mobilenetv2'):
            weights_cfg = str(self.config.get('keras_weights', 'none')).lower().strip()
            if weights_cfg in ('none', 'null', ''):
                weights = 'imagenet' if pretrained else None
            else:
                weights = weights_cfg
            base = tf.keras.applications.MobileNetV2(include_top=False, weights=weights, input_shape=input_shape)
            x = layers.GlobalAveragePooling2D()(base.output)
            x = layers.Dense(256, activation='relu')(x)
            out = layers.Dense(num_classes, activation='softmax')(x)
            m = k_models.Model(inputs=base.input, outputs=out)
            m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=keras_lr),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return m

        raise ValueError(f"Unsupported model_type for Keras: {model_type}")

    def _build_vgg16_bn_keras(self, input_shape: Tuple[int, int, int] = (32, 32, 3), 
                             num_classes: int = 10) -> tf.keras.Model:
        img_input = layers.Input(shape=input_shape)
        init = 'he_normal'
        def conv_block(x, filters, num_convs, name_prefix):
            for i in range(num_convs):
                x = layers.Conv2D(filters, 3, padding='same', kernel_initializer=init, name=f"{name_prefix}_conv{i+1}")(x)
                x = layers.BatchNormalization(name=f"{name_prefix}_bn{i+1}")(x)
                x = layers.Activation('relu', name=f"{name_prefix}_relu{i+1}")(x)
            x = layers.MaxPooling2D(2, strides=2, name=f"{name_prefix}_pool")(x)
            return x
        x = conv_block(img_input, 64, 2, 'block1')
        x = conv_block(x, 128, 2, 'block2')
        x = conv_block(x, 256, 3, 'block3')
        x = conv_block(x, 512, 3, 'block4')
        x = conv_block(x, 512, 3, 'block5')
        x = layers.GlobalAveragePooling2D(name='avgpool')(x)
        x = layers.Dense(512, activation='relu', kernel_initializer=init, name='fc1')(x)
        x = layers.Dense(num_classes, activation='softmax', kernel_initializer=init, name='predictions')(x)
        return k_models.Model(img_input, x, name='vgg16_bn')

    def _build_resnet_cifar_keras(
        self,
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        num_classes: int = 10,
        blocks: Tuple[int, int, int, int] = (2, 2, 2, 2),
        name: str = "resnet18_cifar",
    ) -> tf.keras.Model:
        """Builds a CIFAR-style ResNet with basic residual blocks."""

        def basic_block(x, filters: int, stride: int, block_name: str):
            shortcut = x

            y = layers.Conv2D(
                filters,
                kernel_size=3,
                strides=stride,
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal',
                name=f"{block_name}_conv1",
            )(x)
            y = layers.BatchNormalization(name=f"{block_name}_bn1")(y)
            y = layers.Activation('relu', name=f"{block_name}_relu1")(y)

            y = layers.Conv2D(
                filters,
                kernel_size=3,
                strides=1,
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal',
                name=f"{block_name}_conv2",
            )(y)
            y = layers.BatchNormalization(name=f"{block_name}_bn2")(y)

            in_ch = int(shortcut.shape[-1]) if shortcut.shape[-1] is not None else None
            if stride != 1 or (in_ch is not None and in_ch != filters):
                shortcut = layers.Conv2D(
                    filters,
                    kernel_size=1,
                    strides=stride,
                    padding='same',
                    use_bias=False,
                    kernel_initializer='he_normal',
                    name=f"{block_name}_proj_conv",
                )(shortcut)
                shortcut = layers.BatchNormalization(name=f"{block_name}_proj_bn")(shortcut)

            out = layers.Add(name=f"{block_name}_add")([y, shortcut])
            out = layers.Activation('relu', name=f"{block_name}_out")(out)
            return out

        img_input = layers.Input(shape=input_shape)
        x = layers.Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            name='stem_conv',
        )(img_input)
        x = layers.BatchNormalization(name='stem_bn')(x)
        x = layers.Activation('relu', name='stem_relu')(x)

        stage_filters = [64, 128, 256, 512]
        for si, (num_blocks, filters) in enumerate(zip(blocks, stage_filters), start=1):
            for bi in range(int(num_blocks)):
                stride = 2 if (si > 1 and bi == 0) else 1
                x = basic_block(
                    x,
                    filters=filters,
                    stride=stride,
                    block_name=f"stage{si}_block{bi+1}",
                )

        x = layers.GlobalAveragePooling2D(name='avgpool')(x)
        x = layers.Dense(
            num_classes,
            activation='softmax',
            kernel_initializer='he_normal',
            name='classifier',
        )(x)
        return k_models.Model(img_input, x, name=name)

    def apply_surgery(self, model: tf.keras.Model, masks: Dict[str, np.ndarray]) -> tf.keras.Model:
        """Advanced Functional Rebuilder for Keras with branching support."""
        try:
            def _safe_indices(idx: Any, size: int, fallback: int = 0) -> np.ndarray:
                size = int(size or 0)
                if size <= 0:
                    return np.zeros((0,), dtype=np.int64)
                if idx is None:
                    return np.array([min(max(int(fallback), 0), size - 1)], dtype=np.int64)
                arr = np.asarray(idx).reshape(-1)
                cleaned = []
                for v in arr:
                    try:
                        vv = int(v)
                    except Exception:
                        continue
                    if 0 <= vv < size:
                        cleaned.append(vv)
                if not cleaned:
                    return np.array([min(max(int(fallback), 0), size - 1)], dtype=np.int64)
                return np.unique(np.asarray(cleaned, dtype=np.int64))

            def _normalize_mask(mask: Any, size: int) -> np.ndarray:
                size = int(size or 0)
                if size <= 0:
                    return np.zeros((0,), dtype=bool)
                if mask is None:
                    out = np.ones((size,), dtype=bool)
                    out[0] = True
                    return out
                m = np.asarray(mask).astype(bool).reshape(-1)
                if m.size < size:
                    # Conservative default: unknown entries are kept.
                    m = np.pad(m, (0, size - m.size), constant_values=True)
                elif m.size > size:
                    m = m[:size]
                if not np.any(m):
                    m[0] = True
                return m

            def _producer_layer(tensor):
                kh = tensor._keras_history
                return getattr(kh, "operation", getattr(kh, "layer", None))

            # 1. Harmonize masks across clusters
            pruner = KerasStructuralPruner(model)
            h_masks = pruner.harmonize_masks(masks)
            
            # 2. Rebuild the model using a topological pass
            network_map = {} # old_tensor_id -> new_tensor
            keep_map = {}    # old_tensor_id -> keep_indices
            
            in_shape = model.input_shape[1:]
            if isinstance(in_shape[0], (list, tuple)): # Multiple inputs
                inputs = [tf.keras.Input(shape=s[1:]) for s in in_shape]
                for old_t, new_t in zip(model.inputs, inputs):
                    network_map[id(old_t)] = new_t
                    keep_map[id(old_t)] = np.arange(old_t.shape[-1], dtype=int)
            else:
                inputs = tf.keras.Input(shape=in_shape)
                input_tensor = model.inputs[0]
                network_map[id(input_tensor)] = inputs
                keep_map[id(input_tensor)] = np.arange(input_tensor.shape[-1], dtype=int)
            
            for layer in model.layers:
                if isinstance(layer, layers.InputLayer): continue
                
                # Retrieve inbound tensors and their keep indices
                inbound_t = layer.input
                if isinstance(inbound_t, list):
                    in_tensors = [network_map[id(t)] for t in inbound_t]
                    if isinstance(layer, layers.Concatenate):
                        # For Concatenate, we must concatenate the keep-masks with offsets
                        combined_keep = []
                        offset = 0
                        for t in inbound_t:
                            t_ch = int(t.shape[-1]) if t.shape[-1] is not None else 0
                            k = _safe_indices(keep_map.get(id(t), np.arange(max(t_ch, 1), dtype=np.int64)), max(t_ch, 1))
                            combined_keep.append(k + offset)
                            offset += max(t_ch, 1)
                        in_keep = np.concatenate(combined_keep) if combined_keep else np.array([0], dtype=np.int64)
                    else:
                        # For Add-family merges, prefer common valid indices across branches.
                        b0 = inbound_t[0]
                        ch0 = int(b0.shape[-1]) if b0.shape[-1] is not None else 0
                        base_keep = _safe_indices(keep_map.get(id(b0), np.arange(max(ch0, 1), dtype=np.int64)), max(ch0, 1))
                        for bt in inbound_t[1:]:
                            chb = int(bt.shape[-1]) if bt.shape[-1] is not None else 0
                            kb = _safe_indices(keep_map.get(id(bt), np.arange(max(chb, 1), dtype=np.int64)), max(chb, 1))
                            if chb == ch0:
                                inter = np.intersect1d(base_keep, kb)
                                if inter.size > 0:
                                    base_keep = inter
                        in_keep = base_keep
                else:
                    in_tensors = network_map[id(inbound_t)]
                    in_ch = int(inbound_t.shape[-1]) if inbound_t.shape[-1] is not None else 0
                    in_keep = _safe_indices(keep_map.get(id(inbound_t), np.arange(max(in_ch, 1), dtype=np.int64)), max(in_ch, 1))
                
                cfg = layer.get_config().copy()
                curr_keep = in_keep # Default: same as input
                
                if isinstance(layer, layers.Conv2D):
                    w = layer.get_weights()
                    if not w:
                        raise ValueError(f"Layer '{layer.name}' has no weights for Conv2D surgery.")
                    in_ch = int(w[0].shape[2])
                    out_ch = int(w[0].shape[3])
                    in_keep_eff = _safe_indices(in_keep, in_ch)
                    mask = _normalize_mask(h_masks.get(layer.name, np.ones(out_ch, dtype=bool)), out_ch)
                    keep_out = np.where(mask)[0].astype(np.int64)
                    
                    cfg['filters'] = int(len(keep_out))
                    new_layer = layers.Conv2D.from_config(cfg)
                    x = new_layer(in_tensors)
                    
                    # Slice InChannels (axis 2) and OutChannels (axis 3)
                    k = w[0][:, :, in_keep_eff, :][:, :, :, keep_out]
                    new_w = [k]
                    if len(w) > 1: new_w.append(w[1][keep_out])
                    new_layer.set_weights(new_w)
                    curr_keep = keep_out.astype(np.int64)
                    
                elif isinstance(layer, layers.BatchNormalization):
                    w = layer.get_weights()
                    if not w:
                        new_layer = layers.BatchNormalization.from_config(cfg)
                        x = new_layer(in_tensors)
                        curr_keep = in_keep
                    else:
                        feat = int(w[0].shape[0])
                        bn_keep = _safe_indices(in_keep, feat)
                        new_layer = layers.BatchNormalization.from_config(cfg)
                        x = new_layer(in_tensors)
                        new_layer.set_weights([v[bn_keep] for v in w])
                        curr_keep = bn_keep
                    
                elif isinstance(layer, layers.Dense):
                    new_layer = layers.Dense.from_config(cfg)
                    x = new_layer(in_tensors)
                    w = layer.get_weights()
                    in_f = w[0].shape[0]
                    in_keep_arr = np.asarray(in_keep, dtype=np.int64).reshape(-1)

                    # Case A: direct channel indexing fits Dense input width (e.g. GAP -> Dense).
                    if in_keep_arr.size > 0 and np.all((in_keep_arr >= 0) & (in_keep_arr < in_f)):
                        new_idx = np.unique(in_keep_arr)
                    else:
                        new_idx = np.array([], dtype=np.int64)

                    # Case B: flattened feature map -> Dense, expand channel ids to flattened indices.
                    if new_idx.size == 0:
                        try:
                            p = _producer_layer(inbound_t if not isinstance(inbound_t, list) else inbound_t[0])
                        except Exception:
                            p = None
                        if isinstance(p, layers.Flatten):
                            p_in = p.input
                            prev_c = int(p_in.shape[-1]) if p_in.shape[-1] is not None else 0
                            if prev_c > 0 and in_f % prev_c == 0:
                                base = _safe_indices(in_keep_arr, prev_c)
                                spatial = int(in_f // prev_c)
                                expanded = np.concatenate([base * spatial + s for s in range(spatial)])
                                new_idx = _safe_indices(expanded, in_f)

                    if new_idx.size == 0:
                        new_idx = _safe_indices(in_keep_arr, in_f)

                    k = w[0][new_idx, :]
                    new_layer.set_weights([k, w[1]])
                    curr_keep = np.arange(layer.units, dtype=np.int64)
                else:
                    new_layer = layer.__class__.from_config(cfg)
                    x = new_layer(in_tensors)
                    if layer.get_weights(): 
                        # Handle potential input size changes for layers with weights
                        try:
                            new_layer.set_weights(layer.get_weights())
                        except Exception:
                            # If weights fail to set, it's likely a dimension mismatch
                            pass
                    # For shape-preserving ops (ReLU/Pool/etc), propagate valid keep indices.
                    out_ch = None
                    try:
                        if not isinstance(layer.output, list):
                            out_ch = int(layer.output.shape[-1]) if layer.output.shape[-1] is not None else None
                    except Exception:
                        out_ch = None
                    curr_keep = _safe_indices(in_keep, out_ch) if out_ch else np.asarray(in_keep, dtype=np.int64)

                # Map output tensors
                if isinstance(layer.output, list):
                    for old_t, new_t in zip(layer.output, x):
                        network_map[id(old_t)] = new_t
                        keep_map[id(old_t)] = curr_keep
                else:
                    network_map[id(layer.output)] = x
                    keep_map[id(layer.output)] = curr_keep
            
            output_tensor = model.outputs[0]
            new_model = tf.keras.Model(inputs=inputs, outputs=network_map[id(output_tensor)])
            new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(self.config.get('lr', 3e-4))),
                              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return new_model
        except Exception as e:
            raise SurgeryError(f"Keras structural rebuild failed: {e}")

    def _eval_loss_acc(self, model: tf.keras.Model, loader: Any) -> Tuple[Optional[float], Optional[float]]:
        if loader is None: return None, None
        eval_loader, eval_kwargs = self._prepare_eval_loader(model, loader)
        vals = model.evaluate(eval_loader, verbose=0, **eval_kwargs)
        if isinstance(vals, (list, tuple)) and len(vals) >= 2:
            return float(vals[0]), 100.0 * float(vals[1])
        return None, None

    @timer
    def train(self, model: tf.keras.Model, loader: Any, epochs: int, name: str, 
              val_loader: Any = None, plot: bool = True) -> dict:
        name = str(name or "train")
        auto_baseline = bool((self.config or {}).get("baseline_checkpoint_policy", "auto") != "off")
        if auto_baseline and self._is_baseline_name(name):
            latest = self._latest_baseline_ckpt(model)
            if latest is not None and latest.exists() and bool((self.config or {}).get("baseline_prefer_existing", True)):
                self.load_checkpoint(model, str(latest))
                print(f"[baseline] Loaded existing checkpoint: {latest}")
                hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
                return hist

        class ColabLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                msg = f"Epoch {epoch+1}/{epochs} - loss: {logs.get('loss', 0):.4f} - acc: {logs.get('accuracy', 0):.4f}"
                if 'val_loss' in logs:
                    msg += f" - val_loss: {logs.get('val_loss', 0):.4f} - val_acc: {logs.get('val_accuracy', 0):.4f}"
                print(f"[train] {msg}")

        fit_loader, fit_kwargs = self._prepare_fit_loader(model, loader, is_validation=False)
        val_data, val_kwargs = self._prepare_fit_loader(model, val_loader, is_validation=True) if val_loader else (None, {})

        callbacks = [ColabLogger()]
        if val_loader is not None:
            if bool(self.config.get("keras_reduce_lr_on_plateau", True)):
                callbacks.append(
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_accuracy",
                        mode="max",
                        factor=float(self.config.get("keras_lr_reduce_factor", 0.5)),
                        patience=max(1, int(self.config.get("keras_lr_patience", 1))),
                        min_lr=float(self.config.get("keras_min_lr", 1e-6)),
                        verbose=1,
                    )
                )
            if bool(self.config.get("keras_restore_best", True)):
                callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_accuracy",
                        mode="max",
                        patience=max(1, int(self.config.get("keras_early_stop_patience", 2))),
                        restore_best_weights=True,
                        verbose=1,
                    )
                )

        h = model.fit(
            fit_loader,
            validation_data=val_data,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            **fit_kwargs,
            **val_kwargs,
        )
        hist = {
            'train_loss': list(h.history.get('loss', [])),
            'train_acc': [100.0 * float(x) for x in h.history.get('accuracy', [])],
            'val_loss': list(h.history.get('val_loss', [])),
            'val_acc': [100.0 * float(x) for x in h.history.get('val_accuracy', [])],
        }
        if plot:
            try:
                from ..visualization.stakeholder import plot_training_history
                plot_training_history(hist, title=name)
            except Exception: pass

        if auto_baseline and self._is_baseline_name(name):
            out = self._new_baseline_ckpt(model)
            self.save_checkpoint(model, str(out))
            print(f"[baseline] Saved checkpoint: {out}")

        return hist

    def evaluate(self, model: tf.keras.Model, loader: Any) -> float:
        eval_loader, eval_kwargs = self._prepare_eval_loader(model, loader)
        return model.evaluate(eval_loader, verbose=0, **eval_kwargs)[1] * 100

    def get_viz_data(self, model: tf.keras.Model, loader: Any, num_layers: int = 3) -> dict:
        conv_layers = [l for l in model.layers if isinstance(l, layers.Conv2D)]
        model_input = model.inputs[0]
        viz_model = k_models.Model(inputs=model_input, outputs=[l.output for l in conv_layers[:num_layers]])
        wrapped_loader = KerasLoaderWrapper(loader, model.input_shape[1:])
        x, _ = next(iter(wrapped_loader))
        x = x[:1]
        outputs = viz_model.predict(x, verbose=0)
        if num_layers == 1: outputs = [outputs]
        return {
            "activations": [o[0].transpose(2, 0, 1) for o in outputs],
            "filters": conv_layers[0].get_weights()[0].transpose(3, 2, 0, 1)
        }

    def _estimate_flops(self, model: tf.keras.Model) -> float:
        flops = 0.0
        for l in model.layers:
            if isinstance(l, layers.Conv2D):
                try:
                    w = l.get_weights()[0]
                    kh, kw, cin, cout = w.shape
                    out_shape = l.output_shape
                    h, w_out = out_shape[1], out_shape[2]
                    flops += 2.0 * h * w_out * cin * cout * kh * kw
                except Exception: pass
            elif isinstance(l, layers.Dense):
                try:
                    w = l.get_weights()[0]
                    cin, cout = w.shape
                    flops += 2.0 * cin * cout
                except Exception: pass
        return float(flops)

    def get_stats(self, model: tf.keras.Model, loader: Any = None) -> Tuple[float, float]:
        est = self._estimate_flops(model)
        params = float(model.count_params())
        return est, params

    def save_checkpoint(self, model: tf.keras.Model, path: str):
        model.save_weights(path)

    def load_checkpoint(self, model: tf.keras.Model, path: str):
        model.load_weights(path)

    def get_score_map(self, model: tf.keras.Model, loader: Any, method: str) -> Dict[str, np.ndarray]:
        method = method.lower().strip()
        if method in ("mean_abs_act", "apoz"):
            return self._activation_scores(model, loader, method)
        conv_layers = [(l.name, l) for l in model.layers if isinstance(l, layers.Conv2D)]
        tools = CustomMethodTools(
            framework="keras",
            model=model,
            loader=loader,
            config=self.config,
            prunables=conv_layers,
        )
        score_map = {}
        for name, layer in conv_layers:
            fn_kwargs = {
                "layer_name": name, "layer": layer, "model": model, "loader": loader,
                "tools": tools, "prunables": conv_layers
            }
            for k, v in (self.config or {}).items():
                fn_kwargs.setdefault(k, v)
            s = call_score_fn(method, "keras", fn_kwargs)
            if s is None: raise ValueError(f"Method {method} returned None for layer {name}")
            score_map[name] = np.asarray(s).reshape(-1)
        return score_map

    def _activation_scores(self, model: tf.keras.Model, loader: Any,
                           method: str) -> Dict[str, np.ndarray]:
        conv_layers = [(l.name, l) for l in model.layers if isinstance(l, layers.Conv2D)]
        if not conv_layers: return {}
        model_input = model.inputs[0]
        out_tensors = [l.output for _, l in conv_layers]
        probe_model = k_models.Model(inputs=model_input, outputs=out_tensors)
        acc = {n: np.zeros(l.get_weights()[0].shape[-1], dtype=np.float64) for n, l in conv_layers}
        cnt = {n: 0 for n, _ in conv_layers}
        
        wrapped_loader = KerasLoaderWrapper(loader, model.input_shape[1:])
        batches = self._resolve_prune_batches(wrapped_loader)
        it = iter(wrapped_loader)
        for bi in range(batches):
            try: 
                batch = next(it)
                if isinstance(batch, (list, tuple)): x, y = batch[0], batch[1]
                else: x = batch
            except StopIteration: break
            
            outs = probe_model(x, training=False)
            if len(conv_layers) == 1:
                outs = [outs]
            for (name, _), a in zip(conv_layers, outs):
                a_np = a.numpy()
                elems = a_np.shape[0] * a_np.shape[1] * a_np.shape[2]
                if method == "mean_abs_act":
                    acc[name] += np.abs(a_np).sum(axis=(0, 1, 2))
                elif method == "apoz":
                    # APoZ is defined on post-ReLU activations.
                    post_relu = np.maximum(a_np, 0.0)
                    acc[name] += (post_relu == 0).sum(axis=(0, 1, 2))
                cnt[name] += elems
        score_map = {}
        for name, _ in conv_layers:
            c = max(cnt[name], 1)
            if method == "mean_abs_act":
                s = acc[name] / c
            else:
                s = 1.0 - (acc[name] / c)
            score_map[name] = np.asarray(s, dtype=np.float64).reshape(-1)
        return score_map
