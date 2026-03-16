import numpy as np
from typing import Dict, Any, Callable, Tuple
import os
import copy

from ..core.adapter import FrameworkAdapter
from ..core.decorators import timer
from ..core.exceptions import SurgeryError
from ..pruner.registry import get_method, call_score_fn

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models as k_models
except ImportError:
    pass

class KerasAdapter(FrameworkAdapter):
    def __init__(self, config: dict = None):
        self.config = config or {}

    def get_model(self, model_type: str):
        model_type = str(model_type).lower().strip()
        keras_lr = float(self.config.get('keras_lr', self.config.get('lr', 3e-4)))
        
        if model_type == 'vgg16':
            m = self._build_vgg16_bn_keras(input_shape=(32, 32, 3), num_classes=10)
            m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=keras_lr),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return m

        if model_type in ('resnet50', 'resnet'):
            weights_cfg = str(self.config.get('keras_weights', 'none')).lower().strip()
            weights = None if weights_cfg in ('none', 'null', '') else weights_cfg
            
            base = tf.keras.applications.ResNet50(include_top=False, weights=weights, input_shape=(32, 32, 3))
            x = layers.GlobalAveragePooling2D()(base.output)
            x = layers.Dense(512, activation='relu')(x)
            out = layers.Dense(10, activation='softmax')(x)
            m = k_models.Model(inputs=base.input, outputs=out)
            m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=keras_lr),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return m

        raise ValueError(f"Unsupported model_type for Keras: {model_type}")

    def _build_vgg16_bn_keras(self, input_shape=(32, 32, 3), num_classes=10):
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

    def _eval_loss_acc(self, model, loader):
        if loader is None: return None, None
        vals = model.evaluate(loader, verbose=0)
        if isinstance(vals, (list, tuple)) and len(vals) >= 2:
            return float(vals[0]), 100.0 * float(vals[1])
        return None, None

    @timer
    def train(self, model, loader, epochs, name, val_loader=None, plot=True):
        # Custom Callback for explicit logging in Colab
        class ColabLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                msg = f"Epoch {epoch+1}/{epochs} - loss: {logs.get('loss', 0):.4f} - acc: {logs.get('accuracy', logs.get('acc', 0)):.4f}"
                if 'val_loss' in logs:
                    msg += f" - val_loss: {logs.get('val_loss', 0):.4f} - val_acc: {logs.get('val_accuracy', logs.get('val_acc', 0)):.4f}"
                print(f"📊 {msg}")

        h = model.fit(loader, validation_data=val_loader, epochs=epochs, 
                      verbose=1, callbacks=[ColabLogger()])
        
        hist = {
            'train_loss': list(h.history.get('loss', [])),
            'train_acc': [100.0 * float(x) for x in h.history.get('accuracy', h.history.get('acc', []))],
            'val_loss': list(h.history.get('val_loss', [])),
            'val_acc': [100.0 * float(x) for x in h.history.get('val_accuracy', h.history.get('val_acc', []))],
        }

        if plot:
            try:
                from ..visualization.stakeholder import plot_training_history
                plot_training_history(hist, title=name)
            except Exception as e:
                print(f"⚠️ Could not plot history: {e}")

        return hist

    def evaluate(self, model, loader):
        return model.evaluate(loader, verbose=0)[1] * 100

    def get_viz_data(self, model, loader, num_layers=3):
        conv_layers = [l for l in model.layers if isinstance(l, layers.Conv2D)]
        model_input = model.inputs[0] if getattr(model, "inputs", None) else model.input
        viz_model = k_models.Model(inputs=model_input, outputs=[l.output for l in conv_layers[:num_layers]])
        x, _ = next(iter(loader)); x = x[:1]
        outputs = viz_model.predict(x, verbose=0)
        if num_layers == 1: outputs = [outputs]
        return {
            "activations": [o[0].transpose(2, 0, 1) for o in outputs],
            "filters": conv_layers[0].get_weights()[0].transpose(3, 2, 0, 1)
        }

    def _estimate_flops(self, model):
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

    def get_stats(self, model):
        est = self._estimate_flops(model)
        params = float(model.count_params())
        return est, params

    def save_checkpoint(self, model, path):
        model.save_weights(path)

    def load_checkpoint(self, model, path):
        model.load_weights(path)

    def get_score_map(self, model, loader, method: str) -> Dict[str, np.ndarray]:
        method = method.lower().strip()
        if method in ("taylor", "mean_abs_act", "apoz", "variance_act"):
            return self._activation_or_taylor_scores(model, loader, method)
            
        conv_layers = [(l.name, l) for l in model.layers if isinstance(l, layers.Conv2D)]
        score_map = {}
        for name, layer in conv_layers:
            s = call_score_fn(method, {
                "layer_name": name,
                "layer": layer,
                "model": model,
                "loader": loader,
            })
            if s is None:
                raise ValueError(f"Method {method} returned None for layer {name}")
            score_map[name] = np.asarray(s).reshape(-1)
        return score_map

    def _activation_or_taylor_scores(self, model, loader, method: str) -> Dict[str, np.ndarray]:
        conv_layers = [(l.name, l) for l in model.layers if isinstance(l, layers.Conv2D)]
        if not conv_layers: return {}

        batches = int(self.config.get("prune_batches", 5))
        batches = max(1, batches)

        model_input = model.inputs[0] if getattr(model, "inputs", None) else model.input
        out_tensors = [l.output for _, l in conv_layers] + [model.output]
        probe_model = k_models.Model(inputs=model_input, outputs=out_tensors)

        acc = {n: np.zeros(l.get_weights()[0].shape[-1], dtype=np.float64) for n, l in conv_layers}
        cnt = {n: 0 for n, _ in conv_layers}
        sum1 = {n: np.zeros_like(acc[n]) for n, _ in conv_layers}
        sum2 = {n: np.zeros_like(acc[n]) for n, _ in conv_layers}

        it = iter(loader)
        for bi in range(batches):
            try: x, y = next(it)
            except StopIteration: break

            if method == "taylor":
                with tf.GradientTape() as tape:
                    outs = probe_model(x, training=False)
                    conv_outs = outs[:-1]
                    logits = outs[-1]
                    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits))
                grads = tape.gradient(loss, conv_outs)
                for (name, _), a, g in zip(conv_layers, conv_outs, grads):
                    acc[name] += np.abs(a.numpy() * g.numpy()).sum(axis=(0, 1, 2))
                    cnt[name] += a.shape[0] * a.shape[1] * a.shape[2]
            else:
                outs = probe_model(x, training=False)
                for (name, _), a in zip(conv_layers, outs[:-1]):
                    a_np = a.numpy()
                    elems = a_np.shape[0] * a_np.shape[1] * a_np.shape[2]
                    if method == "mean_abs_act": acc[name] += np.abs(a_np).sum(axis=(0, 1, 2))
                    elif method == "apoz": acc[name] += (a_np == 0).sum(axis=(0, 1, 2))
                    elif method == "variance_act":
                        sum1[name] += a_np.sum(axis=(0, 1, 2))
                        sum2[name] += (a_np ** 2).sum(axis=(0, 1, 2))
                    cnt[name] += elems

        score_map = {}
        for name, _ in conv_layers:
            c = max(cnt[name], 1)
            if method == "mean_abs_act": s = acc[name] / c
            elif method == "apoz": s = 1.0 - (acc[name] / c)
            elif method == "variance_act":
                mean = sum1[name] / c
                s = np.maximum((sum2[name] / c) - (mean ** 2), 0.0)
            elif method == "taylor": s = acc[name] / c
            score_map[name] = np.asarray(s, dtype=np.float64).reshape(-1)
        return score_map

    def apply_surgery(self, model, masks: Dict[str, np.ndarray]):
        try:
            inputs = tf.keras.Input(shape=model.input_shape[1:])
            x = inputs
            prev_keep = np.arange(model.input_shape[-1], dtype=int)
            
            for layer in model.layers:
                if isinstance(layer, layers.InputLayer): continue
                
                cfg = layer.get_config().copy()
                if isinstance(layer, layers.Conv2D):
                    mask = np.asarray(masks.get(layer.name, np.ones(layer.filters, dtype=bool))).astype(bool)
                    keep_out = np.where(mask)[0]
                    if keep_out.size == 0: keep_out = np.array([0], dtype=int)
                    
                    cfg['filters'] = int(len(keep_out))
                    new_layer = layers.Conv2D.from_config(cfg)
                    x = new_layer(x)
                    
                    w = layer.get_weights()
                    k = w[0][:, :, prev_keep, :][:, :, :, keep_out]
                    new_w = [k]
                    if len(w) > 1: new_w.append(w[1][keep_out])
                    new_layer.set_weights(new_w)
                    prev_keep = keep_out
                    
                elif isinstance(layer, layers.BatchNormalization):
                    new_layer = layers.BatchNormalization.from_config(cfg)
                    x = new_layer(x)
                    w = layer.get_weights()
                    new_w = [v[prev_keep] for v in w]
                    new_layer.set_weights(new_w)
                    
                elif isinstance(layer, layers.Dense):
                    new_layer = layers.Dense.from_config(cfg)
                    x = new_layer(x)
                    w = layer.get_weights()
                    k = w[0][prev_keep, :]
                    new_layer.set_weights([k, w[1]])
                    prev_keep = np.arange(layer.units, dtype=int)
                    
                else:
                    new_layer = layer.__class__.from_config(cfg)
                    x = new_layer(x)
                    if layer.get_weights():
                        new_layer.set_weights(layer.get_weights())
            
            new_model = tf.keras.Model(inputs=inputs, outputs=x, name=f"{model.name}_pruned")
            new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(self.config.get('lr', 3e-4))),
                              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return new_model
        except Exception as e:
            raise SurgeryError(f"Keras structural rebuild failed: {e}")
