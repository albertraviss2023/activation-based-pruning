import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


class CustomMethodTools:
    """Backend-provided utilities for custom pruning methods.

    Goal:
    - Keep registration notebooks focused on method math.
    - Move data collection, caching, and cross-layer bookkeeping into package code.
    """

    def __init__(
        self,
        framework: str,
        model: Any,
        loader: Any,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        prunables: Optional[List[Tuple[str, Any]]] = None,
    ):
        self.framework = str(framework).lower().strip()
        self.model = model
        self.loader = loader
        self.device = device or "cpu"
        self.config = config or {}
        self._prunables = prunables
        self._cache: Dict[Any, Any] = {}

    @staticmethod
    def entropy_1d(vals: np.ndarray, bins: int = 24, eps: float = 1e-12) -> float:
        vals = np.asarray(vals, dtype=np.float64).reshape(-1)
        if vals.size == 0:
            return 0.0
        h, _ = np.histogram(vals, bins=bins)
        p = h.astype(np.float64)
        p = p / max(float(p.sum()), 1.0)
        return float(-(p * np.log(p + eps)).sum())

    @staticmethod
    def class_entropy_discriminability(class_channel_matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        m = np.asarray(class_channel_matrix, dtype=np.float64)
        if m.ndim != 2:
            return np.ones((m.shape[-1],), dtype=np.float64)
        k = m.shape[0]
        if k <= 1:
            return np.ones((m.shape[1],), dtype=np.float64)
        p = m / (np.sum(m, axis=0, keepdims=True) + eps)
        ent = -np.sum(p * np.log(p + eps), axis=0) / np.log(float(k) + eps)
        return np.asarray(1.0 - ent, dtype=np.float64)

    @staticmethod
    def tis_threshold_aggregate(class_channel_matrix: np.ndarray, percentile: float = 75.0) -> np.ndarray:
        m = np.asarray(class_channel_matrix, dtype=np.float64)
        if m.ndim != 2:
            return np.asarray(m).reshape(-1)
        if m.shape[0] == 0:
            return np.zeros((m.shape[1],), dtype=np.float64)
        agg = np.zeros((m.shape[1],), dtype=np.float64)
        hit = np.zeros((m.shape[1],), dtype=np.float64)
        for row in m:
            tau = float(np.percentile(row, percentile))
            keep = (row >= tau).astype(np.float64)
            agg += keep * row
            hit += keep
        return agg + 0.05 * hit

    def _max_batches(self, max_batches: Optional[int]) -> int:
        """Resolves calibration pass length for custom method helpers.

        Policy:
        - Explicit method argument wins.
        - Then explicit config override.
        - Else default to full loader length when available.
        - Else use a safe fallback cap for non-sized/infinite iterables.
        """
        if max_batches is not None:
            return max(1, int(max_batches))

        for key in ("prune_batches", "calib_batches", "calibration_batches"):
            val = self.config.get(key, None)
            if val is None or val == "":
                continue
            try:
                iv = int(val)
                if iv > 0:
                    return iv
            except Exception:
                pass

        try:
            n = int(len(self.loader))
            if n > 0:
                return n
        except Exception:
            pass

        return max(1, int(self.config.get("prune_batches_fallback", 128)))

    def _layer_key(self, layer: Any) -> str:
        if hasattr(layer, "name"):
            return str(layer.name)
        return f"id:{id(layer)}"

    def channel_matrix(self, act: np.ndarray) -> np.ndarray:
        a = np.asarray(act)
        if a.ndim == 4:
            if self.framework == "torch":
                return a.transpose(1, 0, 2, 3).reshape(a.shape[1], -1)
            return a.transpose(3, 0, 1, 2).reshape(a.shape[3], -1)
        if a.ndim == 2:
            return a.T
        return a.reshape(a.shape[0], -1).T

    def pooled_nc(self, act: np.ndarray) -> np.ndarray:
        a = np.asarray(act)
        if a.ndim == 4:
            if self.framework == "torch":
                return a.mean(axis=(2, 3))
            return a.mean(axis=(1, 2))
        if a.ndim == 2:
            return a
        return a.reshape(a.shape[0], -1)

    def weight_l2(self, layer: Any, mode: str = "sum", eps: float = 1e-12) -> np.ndarray:
        mode = str(mode).lower().strip()
        if self.framework == "torch":
            w = layer.weight.data.cpu().numpy()
            axes = tuple(range(1, w.ndim))
        else:
            w = layer.get_weights()[0]
            axes = tuple(range(w.ndim - 1))
        if not axes:
            return np.sqrt(np.square(w) + eps)
        if mode == "rms":
            return np.sqrt(np.mean(np.square(w), axis=axes) + eps)
        return np.sqrt(np.sum(np.square(w), axis=axes) + eps)

    def collect_layer_outputs(
        self,
        layer: Any,
        max_batches: Optional[int] = None,
        include_labels: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        key = ("collect", self._layer_key(layer), self._max_batches(max_batches), bool(include_labels))
        if key in self._cache:
            return self._cache[key]

        if self.model is None or self.loader is None:
            return None, None

        max_b = self._max_batches(max_batches)
        if self.framework == "torch":
            import torch

            activations: List[np.ndarray] = []
            labels: List[np.ndarray] = []

            def hook(_m, _i, o):
                if isinstance(o, tuple):
                    o = o[0]
                activations.append(o.detach().cpu().numpy())

            h = layer.register_forward_hook(hook)
            self.model.eval()
            with torch.no_grad():
                for bi, batch in enumerate(self.loader):
                    if bi >= max_b:
                        break
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        x, y = batch[0], batch[1]
                        if include_labels:
                            labels.append(y.detach().cpu().numpy().reshape(-1))
                    else:
                        x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    self.model(x.to(self.device))
            h.remove()

        else:
            import tensorflow as tf

            model_in = self.model.inputs[0] if isinstance(self.model.inputs, (list, tuple)) else self.model.inputs
            probe = tf.keras.Model(inputs=model_in, outputs=layer.output)
            activations = []
            labels = []
            for bi, batch in enumerate(self.loader):
                if bi >= max_b:
                    break
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                    if include_labels:
                        y_np = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
                        labels.append(y_np.reshape(-1))
                else:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                out = probe(x, training=False)
                out_np = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
                activations.append(out_np)

        if not activations:
            out = (None, None)
            self._cache[key] = out
            return out

        A = np.concatenate(activations, axis=0)
        if include_labels and labels:
            Y = np.concatenate(labels, axis=0)
        else:
            Y = np.zeros((A.shape[0],), dtype=np.int64)
        out = (A, Y)
        self._cache[key] = out
        return out

    def rank_scores(self, act: np.ndarray, max_samples: int = 32) -> np.ndarray:
        a = np.asarray(act)
        if a.ndim == 4:
            n = min(a.shape[0], int(max_samples))
            if self.framework == "torch":
                c = a.shape[1]
                return np.asarray(
                    [np.mean([np.linalg.matrix_rank(a[i, j]) for i in range(n)]) for j in range(c)],
                    dtype=np.float64,
                )
            c = a.shape[3]
            return np.asarray(
                [np.mean([np.linalg.matrix_rank(a[i, :, :, j]) for i in range(n)]) for j in range(c)],
                dtype=np.float64,
            )
        x = self.channel_matrix(a)
        return np.sqrt(np.var(x, axis=1) + 1e-12)

    def spectral_energy_scores(self, act: np.ndarray, max_samples: int = 32) -> np.ndarray:
        a = np.asarray(act)
        if a.ndim == 4:
            n = min(a.shape[0], int(max_samples))
            if self.framework == "torch":
                c = a.shape[1]
                vals = []
                for j in range(c):
                    e = []
                    for i in range(n):
                        f = np.fft.fft2(a[i, j])
                        e.append(np.mean(np.abs(f) ** 2))
                    vals.append(np.mean(e))
                return np.asarray(vals, dtype=np.float64)
            c = a.shape[3]
            vals = []
            for j in range(c):
                e = []
                for i in range(n):
                    f = np.fft.fft2(a[i, :, :, j])
                    e.append(np.mean(np.abs(f) ** 2))
                vals.append(np.mean(e))
            return np.asarray(vals, dtype=np.float64)
        x = self.channel_matrix(a)
        return np.mean(np.square(x), axis=1)

    def corr_redundancy_scores(self, act: np.ndarray) -> np.ndarray:
        x = self.channel_matrix(act)
        if x.shape[0] <= 1:
            return np.ones((x.shape[0],), dtype=np.float64)
        r = np.corrcoef(x)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(r, 0.0)
        redundancy = np.mean(np.abs(r), axis=1)
        return np.asarray(1.0 - redundancy, dtype=np.float64)

    @staticmethod
    def chip_nuclear_independence_scores(act: np.ndarray, framework: str, max_spatial: int = 196) -> Optional[np.ndarray]:
        a = np.asarray(act, dtype=np.float64)
        if a.ndim != 4:
            return None
        fw = str(framework).lower().strip()
        if fw == "torch":
            n, c, h, w = a.shape
            flat = a.reshape(n, c, h * w)
        else:
            n, h, w, c = a.shape
            flat = a.transpose(0, 3, 1, 2).reshape(n, c, h * w)
        hw = flat.shape[-1]
        if hw > max_spatial:
            idx = np.linspace(0, hw - 1, num=max_spatial, dtype=np.int64)
            flat = flat[:, :, idx]
        m = flat.transpose(0, 2, 1).reshape(-1, c)
        m = m - m.mean(axis=0, keepdims=True)
        s = np.linalg.svd(m, full_matrices=False, compute_uv=False)
        base_nuc = float(np.sum(s))
        scores = np.zeros((c,), dtype=np.float64)
        for i in range(c):
            m_minus = np.delete(m, i, axis=1)
            if m_minus.shape[1] == 0:
                scores[i] = base_nuc
                continue
            s_minus = np.linalg.svd(m_minus, full_matrices=False, compute_uv=False)
            nuc_minus = float(np.sum(s_minus))
            scores[i] = max(base_nuc - nuc_minus, 0.0)
        return scores

    def chip_scores(self, act: np.ndarray, max_spatial: Optional[int] = None) -> np.ndarray:
        """Backend-level CHIP helper robust to Conv (4D) and Linear/Dense (2D) activations."""
        from .chip import chip_channel_independence_scores

        a = np.asarray(act)
        if a.ndim == 4:
            channel_axis = 1 if self.framework == "torch" else -1
            if max_spatial is None:
                spatial_total = int(a.shape[2] * a.shape[3]) if self.framework == "torch" else int(a.shape[1] * a.shape[2])
                max_spatial = spatial_total
            return chip_channel_independence_scores(a, channel_axis=channel_axis, max_spatial=int(max_spatial))

        # For non-spatial activations, route through generic 2D chip implementation.
        x = self.pooled_nc(a)  # typically (N, C) for dense-like tensors
        return chip_channel_independence_scores(np.asarray(x), channel_axis=1, max_spatial=None)

    def classwise_taylor_matrix(self, layer: Any, max_batches: Optional[int] = None) -> Optional[np.ndarray]:
        max_b = self._max_batches(max_batches)
        key = ("classwise_taylor", self._layer_key(layer), max_b)
        if key in self._cache:
            return self._cache[key]

        if self.model is None or self.loader is None:
            return None

        if self.framework == "torch":
            import torch
            import torch.nn as nn

            cache = {"o": None}

            def hook(_m, _i, o):
                if isinstance(o, tuple):
                    o = o[0]
                o.retain_grad()
                cache["o"] = o

            h = layer.register_forward_hook(hook)
            crit = nn.CrossEntropyLoss()
            per_class = defaultdict(list)
            self.model.eval()
            for bi, batch in enumerate(self.loader):
                if bi >= max_b:
                    break
                if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                    continue
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                self.model.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = crit(logits, y)
                loss.backward()
                o = cache.get("o", None)
                if o is None or o.grad is None:
                    continue
                t = (o.detach() * o.grad.detach()).abs()
                t_ch = t.mean(dim=(2, 3)) if t.dim() == 4 else t
                y_np = y.detach().cpu().numpy().reshape(-1)
                t_np = t_ch.detach().cpu().numpy()
                for cls in np.unique(y_np):
                    idx = np.where(y_np == cls)[0]
                    if idx.size:
                        per_class[int(cls)].append(t_np[idx].mean(axis=0))
            h.remove()
        else:
            import tensorflow as tf

            model_in = self.model.inputs[0] if isinstance(self.model.inputs, (list, tuple)) else self.model.inputs
            model_out = self.model.outputs[0] if isinstance(self.model.outputs, (list, tuple)) else self.model.outputs
            probe = tf.keras.Model(inputs=model_in, outputs=[layer.output, model_out])
            per_class = defaultdict(list)
            for bi, batch in enumerate(self.loader):
                if bi >= max_b:
                    break
                if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                    continue
                x, y = batch[0], batch[1]
                y_t = tf.cast(tf.reshape(y, [-1]), tf.int32)
                with tf.GradientTape() as tape:
                    a, logits = probe(x, training=False)
                    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_t, logits))
                g = tape.gradient(loss, a)
                if g is None:
                    continue
                t = tf.abs(a * g).numpy()
                t_ch = t.mean(axis=(1, 2)) if t.ndim == 4 else t
                y_np = y_t.numpy().reshape(-1)
                for cls in np.unique(y_np):
                    idx = np.where(y_np == cls)[0]
                    if idx.size:
                        per_class[int(cls)].append(t_ch[idx].mean(axis=0))

        if not per_class:
            self._cache[key] = None
            return None
        classes = sorted(per_class.keys())
        mat = np.stack([np.mean(per_class[c], axis=0) for c in classes], axis=0).astype(np.float64)
        self._cache[key] = mat
        return mat

    def taylor_contribution(self, layer: Any, classwise: bool = False, max_batches: Optional[int] = None) -> Optional[np.ndarray]:
        mat = self.classwise_taylor_matrix(layer, max_batches=max_batches)
        if mat is None:
            return None
        if classwise:
            return np.asarray(np.sum(mat, axis=0), dtype=np.float64)
        return np.asarray(np.mean(mat, axis=0), dtype=np.float64)

    def _prunable_layers(self) -> List[Tuple[str, Any]]:
        if self._prunables is not None:
            return list(self._prunables)
        if self.framework == "torch":
            import torch.nn as nn
            return [(n, m) for n, m in self.model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        import tensorflow as tf
        return [(l.name, l) for l in self.model.layers if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.Dense))]

    def _collect_mean_abs_by_layer(self, max_batches: Optional[int] = None) -> Dict[str, np.ndarray]:
        max_b = self._max_batches(max_batches)
        key = ("mean_abs_by_layer", max_b)
        if key in self._cache:
            return self._cache[key]
        pr = self._prunable_layers()
        if not pr:
            self._cache[key] = {}
            return {}

        if self.framework == "torch":
            import torch

            acc: Dict[str, np.ndarray] = {}
            cnt: Dict[str, int] = {}
            hooks = []
            for name, layer in pr:
                ch = int(getattr(layer, "out_channels", getattr(layer, "out_features", 0)))
                acc[name] = np.zeros((max(ch, 1),), dtype=np.float64)
                cnt[name] = 0

                def _mk_hook(layer_name: str):
                    def _hook(_m, _i, o):
                        oo = o[0] if isinstance(o, tuple) else o
                        od = oo.detach()
                        if od.dim() == 4:
                            v = torch.abs(od).mean(dim=(0, 2, 3)).cpu().numpy()
                        else:
                            v = torch.abs(od).mean(dim=0).cpu().numpy()
                        acc[layer_name] += np.asarray(v, dtype=np.float64).reshape(-1)
                        cnt[layer_name] += 1
                    return _hook

                hooks.append(layer.register_forward_hook(_mk_hook(name)))

            self.model.eval()
            with torch.no_grad():
                for bi, batch in enumerate(self.loader):
                    if bi >= max_b:
                        break
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    self.model(x.to(self.device))
            for h in hooks:
                h.remove()

            out = {n: (acc[n] / max(int(cnt.get(n, 0)), 1)).astype(np.float64).reshape(-1) for n, _ in pr}
        else:
            import tensorflow as tf

            model_in = self.model.inputs[0] if isinstance(self.model.inputs, (list, tuple)) else self.model.inputs
            probe = tf.keras.Model(inputs=model_in, outputs=[l.output for _, l in pr])
            acc = {}
            cnt = {}
            for name, layer in pr:
                if hasattr(layer, "filters"):
                    ch = int(layer.filters)
                else:
                    ch = int(layer.units)
                acc[name] = np.zeros((max(ch, 1),), dtype=np.float64)
                cnt[name] = 0
            for bi, batch in enumerate(self.loader):
                if bi >= max_b:
                    break
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                outs = probe(x, training=False)
                if len(pr) == 1:
                    outs = [outs]
                for (name, _), o in zip(pr, outs):
                    a = o.numpy() if hasattr(o, "numpy") else np.asarray(o)
                    if a.ndim == 4:
                        v = np.mean(np.abs(a), axis=(0, 1, 2))
                    else:
                        v = np.mean(np.abs(a), axis=0)
                    acc[name] += np.asarray(v, dtype=np.float64).reshape(-1)
                    cnt[name] += 1
            out = {n: (acc[n] / max(int(cnt.get(n, 0)), 1)).astype(np.float64).reshape(-1) for n, _ in pr}

        self._cache[key] = out
        return out

    @staticmethod
    def _propagate_from_next_torch(next_layer: Any, next_scores: np.ndarray, current_out: int) -> Optional[np.ndarray]:
        if not hasattr(next_layer, "weight"):
            return None
        w = np.abs(next_layer.weight.data.cpu().numpy())
        s_next = np.asarray(next_scores, dtype=np.float64).reshape(-1)
        if w.ndim == 4:
            groups = int(getattr(next_layer, "groups", 1))
            if groups == 1 and w.shape[0] == s_next.size and w.shape[1] == current_out:
                p = np.tensordot(s_next, w, axes=(0, 0))
                return np.asarray(np.sum(p, axis=(1, 2)), dtype=np.float64)
            if groups == current_out and w.shape[1] == 1 and w.shape[0] == s_next.size:
                return np.asarray(s_next * np.sum(w[:, 0, :, :], axis=(1, 2)), dtype=np.float64)
            return None
        if w.ndim == 2 and w.shape[0] == s_next.size:
            vin = np.dot(s_next, w)
            if current_out > 0 and vin.size % current_out == 0:
                return np.asarray(vin.reshape(current_out, -1).sum(axis=1), dtype=np.float64)
            if vin.size == current_out:
                return np.asarray(vin, dtype=np.float64)
        return None

    @staticmethod
    def _propagate_from_next_keras(next_layer: Any, next_scores: np.ndarray, current_out: int) -> Optional[np.ndarray]:
        w = next_layer.get_weights()
        if not w:
            return None
        ww = np.abs(np.asarray(w[0], dtype=np.float64))
        s_next = np.asarray(next_scores, dtype=np.float64).reshape(-1)
        if ww.ndim == 4 and ww.shape[3] == s_next.size and ww.shape[2] == current_out:
            p = np.tensordot(ww, s_next, axes=([3], [0]))
            return np.asarray(np.sum(p, axis=(0, 1)), dtype=np.float64)
        if ww.ndim == 2 and ww.shape[1] == s_next.size:
            vin = np.dot(ww, s_next)
            if current_out > 0 and vin.size % current_out == 0:
                return np.asarray(vin.reshape(current_out, -1).sum(axis=1), dtype=np.float64)
            if vin.size == current_out:
                return np.asarray(vin, dtype=np.float64)
        return None

    def nisp_score_map(self, max_batches: Optional[int] = None) -> Dict[str, np.ndarray]:
        max_b = self._max_batches(max_batches)
        key = ("nisp_score_map", max_b)
        if key in self._cache:
            return self._cache[key]
        pr = self._prunable_layers()
        if not pr:
            self._cache[key] = {}
            return {}
        mean_abs = self._collect_mean_abs_by_layer(max_batches=max_b)
        score_map: Dict[str, np.ndarray] = {}
        last_name, last_layer = pr[-1]
        s_next = np.asarray(mean_abs.get(last_name, self.weight_l2(last_layer, mode="sum")), dtype=np.float64).reshape(-1)
        score_map[last_name] = np.maximum(s_next, 0.0) + 1e-12
        for i in range(len(pr) - 2, -1, -1):
            lname, layer = pr[i]
            _, next_layer = pr[i + 1]
            if self.framework == "torch":
                out_ch = int(getattr(layer, "out_channels", getattr(layer, "out_features", 0)))
                prop = self._propagate_from_next_torch(next_layer, s_next, out_ch)
            else:
                out_ch = int(getattr(layer, "filters", getattr(layer, "units", 0)))
                prop = self._propagate_from_next_keras(next_layer, s_next, out_ch)
            own = np.asarray(mean_abs.get(lname, self.weight_l2(layer, mode="sum")), dtype=np.float64).reshape(-1)
            if prop is None:
                s_curr = own
            else:
                if own.size != prop.size:
                    own = np.resize(own, prop.size)
                s_curr = (0.8 * np.asarray(prop, dtype=np.float64)) + (0.2 * own)
            s_curr = np.maximum(s_curr, 0.0) + 1e-12
            score_map[lname] = s_curr
            s_next = s_curr
        self._cache[key] = score_map
        return score_map

    def thinet_alpha(self, layer: Any) -> np.ndarray:
        pr = self._prunable_layers()
        idx = -1
        for i, (_n, l) in enumerate(pr):
            if l is layer:
                idx = i
                break
        if idx < 0 or idx + 1 >= len(pr):
            return np.asarray(self.weight_l2(layer, mode="sum"), dtype=np.float64).reshape(-1)
        _, nxt = pr[idx + 1]
        if self.framework == "torch":
            out_ch = int(getattr(layer, "out_channels", getattr(layer, "out_features", 0)))
            if not hasattr(nxt, "weight"):
                return np.asarray(self.weight_l2(layer, mode="sum"), dtype=np.float64).reshape(-1)
            w = nxt.weight.data.cpu().numpy()
            if w.ndim == 4 and w.shape[1] == out_ch:
                return np.mean(np.abs(w), axis=(0, 2, 3))
            if w.ndim == 2 and out_ch > 0 and (w.shape[1] % out_ch == 0):
                ww = w.reshape(w.shape[0], out_ch, -1)
                return np.mean(np.abs(ww), axis=(0, 2))
            return np.asarray(self.weight_l2(layer, mode="sum"), dtype=np.float64).reshape(-1)
        out_ch = int(getattr(layer, "filters", getattr(layer, "units", 0)))
        w = nxt.get_weights()
        if not w:
            return np.asarray(self.weight_l2(layer, mode="sum"), dtype=np.float64).reshape(-1)
        nw = w[0]
        if nw.ndim == 4 and nw.shape[2] == out_ch:
            return np.mean(np.abs(nw), axis=(0, 1, 3))
        if nw.ndim == 2 and out_ch > 0 and (nw.shape[0] % out_ch == 0):
            ww = nw.reshape(out_ch, -1, nw.shape[1])
            return np.mean(np.abs(ww), axis=(1, 2))
        return np.asarray(self.weight_l2(layer, mode="sum"), dtype=np.float64).reshape(-1)

    def thinet_reconstruction_scores(self, act: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
        nc = self.pooled_nc(act)
        nc = np.asarray(nc, dtype=np.float64)
        if nc.ndim != 2:
            return np.abs(alpha)
        if nc.shape[1] != alpha.size:
            if nc.shape[1] == 0:
                return np.abs(alpha)
            alpha = np.resize(alpha, nc.shape[1])
        z = nc - nc.mean(axis=0, keepdims=True)
        y = np.dot(z, alpha)
        y = y - np.mean(y)
        yn = np.linalg.norm(y) + 1e-12
        zn = np.linalg.norm(z, axis=0) + 1e-12
        corr = np.abs(np.dot(z.T, y)) / (zn * yn)
        anorm = np.abs(alpha) / (np.max(np.abs(alpha)) + 1e-12)
        return np.asarray(corr * anorm, dtype=np.float64)

    def reprune_representative_scores(self, act: np.ndarray) -> np.ndarray:
        x = self.channel_matrix(act)
        x = np.asarray(x, dtype=np.float64)
        c = x.shape[0]
        if c <= 1:
            return np.ones((c,), dtype=np.float64)
        xc = x - x.mean(axis=1, keepdims=True)
        n = np.linalg.norm(xc, axis=1, keepdims=True) + 1e-12
        xn = xc / n
        sim = np.abs(np.dot(xn, xn.T))
        np.fill_diagonal(sim, 1.0)
        var = np.var(xc, axis=1)
        k = max(1, int(round(np.sqrt(c))))
        reps = [int(np.argmax(var))]
        for _ in range(1, k):
            dist = 1.0 - np.max(sim[:, reps], axis=1)
            dist[reps] = -np.inf
            reps.append(int(np.argmax(dist)))
        reps = np.asarray(sorted(set(reps)), dtype=np.int64)
        assign = np.argmax(sim[:, reps], axis=1)
        sim_to_rep = sim[np.arange(c), reps[assign]]
        scores = 0.2 * (1.0 - sim_to_rep)
        counts = np.bincount(assign, minlength=reps.size).astype(np.float64)
        if counts.size:
            rep_bonus = 1.0 + counts / max(float(c), 1.0)
            for i, r in enumerate(reps):
                scores[r] = rep_bonus[i]
        vnorm = var / (np.max(var) + 1e-12)
        scores = scores + 0.1 * vnorm
        return np.asarray(scores, dtype=np.float64)
