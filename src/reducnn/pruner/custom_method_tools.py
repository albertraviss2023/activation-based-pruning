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
    def tis_threshold_aggregate(class_channel_matrix: np.ndarray, percentile: float = 75.0, eps: float = 1e-12) -> np.ndarray:
        """TIS-style binary class contribution count.

        For each class row, activations/channels above a class threshold receive
        importance 1 and the rest receive 0. The final channel score is the sum
        of class-specific binary contributions. A tiny continuous tie-breaker is
        added so deterministic top-k selection remains stable when many channels
        cover the same number of classes.
        """
        m = np.asarray(class_channel_matrix, dtype=np.float64)
        if m.ndim != 2:
            return np.asarray(m).reshape(-1)
        if m.shape[0] == 0:
            return np.zeros((m.shape[1],), dtype=np.float64)
        hits = np.zeros((m.shape[1],), dtype=np.float64)
        tie = np.zeros((m.shape[1],), dtype=np.float64)
        for row in m:
            tau = float(np.percentile(row, percentile))
            cls_hits = (row >= tau).astype(np.float64)
            hits += cls_hits
            tie += cls_hits * row
        tie = tie / (np.max(np.abs(tie)) + eps)
        return hits + (1e-6 * tie)

    def _max_batches(self, max_batches: Optional[int]) -> int:
        """Resolves calibration pass length for custom method helpers.

        Policy:
        - Explicit method argument wins.
        - Then explicit config override.
        - Else use a bounded default cap, even if loader length is known.
        - This avoids accidental OOM in notebook custom methods that collect
          full intermediate activations (for example CHIP-style methods).
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

        default_cap = max(1, int(self.config.get("prune_batches_default", 8)))
        try:
            n = int(len(self.loader))
            if n > 0:
                return min(n, default_cap)
        except Exception:
            pass

        return max(1, int(self.config.get("prune_batches_fallback", default_cap)))

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
        """CHIP-style nuclear-norm-change channel independence scores.

        CHIP scores a channel by measuring how much the feature-map nuclear norm
        changes when that channel is removed. Larger change means the channel is
        more independent and should be kept.
        """
        from .chip import chip_channel_independence_scores

        a = np.asarray(act)
        if a.ndim == 4:
            if max_spatial is None:
                spatial_total = int(a.shape[2] * a.shape[3]) if self.framework == "torch" else int(a.shape[1] * a.shape[2])
                max_spatial = spatial_total
            nuclear = self.chip_nuclear_independence_scores(a, self.framework, max_spatial=int(max_spatial))
            if nuclear is not None:
                return nuclear

        # Fallback for non-spatial activations.
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
        """NISP-style final-response importance propagation.

        This follows the key NISP recurrence ``s_l = |W_{l+1}|^T s_{l+1}``.
        The final-response layer is initialized from calibration activation
        energy when available, falling back to weight energy.
        """
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
                s_curr = np.asarray(prop, dtype=np.float64)
            s_curr = np.maximum(s_curr, 0.0) + 1e-12
            score_map[lname] = s_curr
            s_next = s_curr
        self._cache[key] = score_map
        return score_map

    def senpis_ablation_scores(self, layer: Any, similarity_threshold: float = 0.90, attenuation_factor: float = 0.5) -> np.ndarray:
        """SeNPIS-style class-wise filter ablation loss delta with attenuation.

        For each class, this computes the absolute loss change caused by zeroing
        one filter/channel, averages across classes, then attenuates redundant
        channels that are highly similar to a stronger channel.
        """
        scores = None
        if self.framework == "torch":
            scores = self._senpis_ablation_scores_torch(layer)
        else:
            scores = self._senpis_ablation_scores_keras(layer)
        if scores is None:
            mat = self.classwise_taylor_matrix(layer)
            if mat is not None:
                scores = np.mean(np.abs(mat), axis=0)
            else:
                scores = self.weight_l2(layer)
        return self._attenuate_redundant_scores(layer, np.asarray(scores, dtype=np.float64), similarity_threshold, attenuation_factor)

    def _senpis_ablation_scores_torch(self, layer: Any) -> Optional[np.ndarray]:
        import torch
        import torch.nn as nn

        if self.model is None or self.loader is None:
            return None
        ch = int(getattr(layer, "out_channels", getattr(layer, "out_features", 0)))
        if ch <= 0:
            return None
        max_b = self._max_batches(None)
        crit = nn.CrossEntropyLoss(reduction="none")
        base_losses: Dict[int, List[float]] = defaultdict(list)
        delta: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros((ch,), dtype=np.float64))
        counts: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros((ch,), dtype=np.float64))

        self.model.eval()
        batches = []
        with torch.no_grad():
            for bi, batch in enumerate(self.loader):
                if bi >= max_b:
                    break
                if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                    continue
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(x)
                losses = crit(logits, y)
                batches.append((x, y, losses.detach()))
                y_np = y.detach().cpu().numpy().reshape(-1)
                for cls in np.unique(y_np):
                    idx = np.where(y_np == cls)[0]
                    if idx.size:
                        base_losses[int(cls)].append(float(losses[idx].mean().item()))

        if not batches:
            return None

        def make_hook(channel: int):
            def _hook(_m, _i, o):
                out = o.clone()
                if out.dim() == 4:
                    out[:, channel, :, :] = 0
                elif out.dim() == 2:
                    out[:, channel] = 0
                return out
            return _hook

        for c in range(ch):
            h = layer.register_forward_hook(make_hook(c))
            with torch.no_grad():
                for x, y, base_batch_loss in batches:
                    logits = self.model(x)
                    losses = crit(logits, y)
                    y_np = y.detach().cpu().numpy().reshape(-1)
                    for cls in np.unique(y_np):
                        idx = np.where(y_np == cls)[0]
                        if idx.size:
                            d = abs(float(losses[idx].mean().item()) - float(base_batch_loss[idx].mean().item()))
                            delta[int(cls)][c] += d
                            counts[int(cls)][c] += 1
            h.remove()

        class_scores = []
        for cls, arr in delta.items():
            class_scores.append(arr / np.maximum(counts[cls], 1.0))
        if not class_scores:
            return None
        return np.mean(np.stack(class_scores, axis=0), axis=0)

    def _senpis_ablation_scores_keras(self, layer: Any) -> Optional[np.ndarray]:
        import tensorflow as tf

        if self.model is None or self.loader is None:
            return None
        ch = int(getattr(layer, "filters", getattr(layer, "units", 0)))
        if ch <= 0:
            return None
        max_b = self._max_batches(None)
        model_in = self.model.inputs[0] if isinstance(self.model.inputs, (list, tuple)) else self.model.inputs
        model_out = self.model.outputs[0] if isinstance(self.model.outputs, (list, tuple)) else self.model.outputs
        layer_out = layer.output
        mask_in = tf.keras.Input(shape=tuple(layer_out.shape[1:]))
        masked = tf.keras.layers.Multiply()([layer_out, mask_in])
        x = masked
        take = False
        for l in self.model.layers:
            if l is layer:
                take = True
                continue
            if not take:
                continue
            try:
                x = l(x)
            except Exception:
                return None
        masked_model = tf.keras.Model(inputs=[model_in, mask_in], outputs=x)
        base_model = tf.keras.Model(inputs=model_in, outputs=model_out)
        delta: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros((ch,), dtype=np.float64))
        counts: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros((ch,), dtype=np.float64))
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")

        for bi, batch in enumerate(self.loader):
            if bi >= max_b:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                continue
            x_batch, y_batch = batch[0], batch[1]
            y_t = tf.cast(tf.reshape(y_batch, [-1]), tf.int32)
            base_logits = base_model(x_batch, training=False)
            base_loss = loss_fn(y_t, base_logits).numpy()
            out_shape = tuple(layer_out.shape[1:])
            mask = np.ones((int(np.asarray(x_batch).shape[0]), *[int(v) for v in out_shape]), dtype=np.float32)
            for c in range(ch):
                m = mask.copy()
                if len(out_shape) == 3:
                    m[..., c] = 0.0
                else:
                    m[:, c] = 0.0
                logits = masked_model([x_batch, m], training=False)
                losses = loss_fn(y_t, logits).numpy()
                y_np = y_t.numpy().reshape(-1)
                for cls in np.unique(y_np):
                    idx = np.where(y_np == cls)[0]
                    if idx.size:
                        delta[int(cls)][c] += abs(float(losses[idx].mean()) - float(base_loss[idx].mean()))
                        counts[int(cls)][c] += 1
        class_scores = [arr / np.maximum(counts[cls], 1.0) for cls, arr in delta.items()]
        if not class_scores:
            return None
        return np.mean(np.stack(class_scores, axis=0), axis=0)

    def _attenuate_redundant_scores(
        self,
        layer: Any,
        scores: np.ndarray,
        similarity_threshold: float,
        attenuation_factor: float,
    ) -> np.ndarray:
        out = np.asarray(scores, dtype=np.float64).reshape(-1).copy()
        sim = self.kernel_similarity_matrix(layer)
        if sim is None or sim.shape[0] != out.size:
            act, _ = self.collect_layer_outputs(layer, include_labels=False)
            if act is not None:
                x = self.channel_matrix(act)
                sim = np.abs(np.corrcoef(x))
                sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
        if sim is None or sim.shape[0] != out.size:
            return out
        np.fill_diagonal(sim, 0.0)
        for i in range(out.size):
            for j in range(i + 1, out.size):
                if sim[i, j] >= similarity_threshold:
                    weaker = i if out[i] <= out[j] else j
                    out[weaker] *= float(attenuation_factor)
        return out

    def kernel_similarity_matrix(self, layer: Any) -> Optional[np.ndarray]:
        w = None
        if self.framework == "torch" and hasattr(layer, "weight"):
            w = layer.weight.data.cpu().numpy()
            if w.ndim >= 2:
                x = w.reshape(w.shape[0], -1)
            else:
                return None
        elif hasattr(layer, "get_weights"):
            ww = layer.get_weights()
            if not ww:
                return None
            w = ww[0]
            if w.ndim == 4:
                x = np.moveaxis(w, -1, 0).reshape(w.shape[-1], -1)
            elif w.ndim == 2:
                x = w.T
            else:
                return None
        else:
            return None
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(axis=1, keepdims=True)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        sim = np.abs(x @ x.T)
        return np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)

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

    def thinet_next_layer_damage_scores(self, layer: Any) -> np.ndarray:
        """ThiNet-style next-layer reconstruction damage proxy.

        Scores the current layer's channels by the mean squared contribution
        they make to the next prunable layer output. This is the per-channel
        marginal form of the next-layer reconstruction objective.
        """
        pr = self._prunable_layers()
        idx = -1
        for i, (_n, l) in enumerate(pr):
            if l is layer:
                idx = i
                break
        if idx < 0 or idx + 1 >= len(pr):
            return np.asarray(self.weight_l2(layer, mode="sum"), dtype=np.float64).reshape(-1)
        _, next_layer = pr[idx + 1]
        act, _ = self.collect_layer_outputs(layer, include_labels=False)
        if act is None:
            return np.asarray(self.weight_l2(layer, mode="sum"), dtype=np.float64).reshape(-1)
        nc = np.asarray(self.pooled_nc(act), dtype=np.float64)
        if nc.ndim != 2 or nc.shape[1] == 0:
            return np.asarray(self.weight_l2(layer, mode="sum"), dtype=np.float64).reshape(-1)
        alpha = self.thinet_alpha(layer).reshape(-1)
        if alpha.size != nc.shape[1]:
            alpha = np.resize(alpha, nc.shape[1])
        contrib = nc * alpha.reshape(1, -1)
        return np.mean(np.square(contrib), axis=0)

    def reprune_representative_scores(self, act: np.ndarray) -> np.ndarray:
        x = self.channel_matrix(act)
        return self.reprune_kernel_coverage_scores_from_matrix(x)

    def reprune_kernel_coverage_scores(self, layer: Any, target_keep_ratio: float = 0.7) -> Optional[np.ndarray]:
        sim = self.kernel_similarity_matrix(layer)
        if sim is None:
            return None
        c = sim.shape[0]
        if c <= 1:
            return np.ones((c,), dtype=np.float64)
        target_clusters = max(1, int(round(c * float(target_keep_ratio))))
        dist = 1.0 - sim
        np.fill_diagonal(dist, np.inf)
        threshold = float(np.partition(dist.reshape(-1), min(dist.size - 1, c * max(c - target_clusters, 1)))[min(dist.size - 1, c * max(c - target_clusters, 1))])
        parent = list(range(c))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(c):
            for j in range(i + 1, c):
                if dist[i, j] <= threshold:
                    union(i, j)
        clusters: Dict[int, List[int]] = defaultdict(list)
        for i in range(c):
            clusters[find(i)].append(i)
        scores = np.zeros((c,), dtype=np.float64)
        for members in clusters.values():
            sub = sim[np.ix_(members, members)]
            rep = members[int(np.argmax(np.mean(sub, axis=1)))]
            scores[rep] += float(len(members))
        return scores + 1e-6

    def reprune_kernel_coverage_scores_from_matrix(self, ch_by_feat: np.ndarray) -> np.ndarray:
        x = np.asarray(ch_by_feat, dtype=np.float64)
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
