import os
import copy
import numpy as np
from typing import Dict, Any, Callable, Tuple
from ..core.adapter import FrameworkAdapter
from ..core.decorators import timer
from ..core.exceptions import SurgeryError
from ..pruner.registry import get_method, call_score_fn

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import models
    from thop import profile
except ImportError:
    pass

class TorchStructuralPruner:
    """V6-style structural channel surgery for Conv/BN/Linear dependencies."""
    def __init__(self, model):
        self.dependencies = self._trace(model)

    def _trace(self, model):
        dep = {}
        convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        names = [n for n, _ in convs]

        for i, (n, m) in enumerate(convs):
            dep[n] = {
                "siblings": [],
                "next": [names[i + 1]] if i + 1 < len(names) else [],
                "out_channels": m.out_channels,
            }
            parts = n.split('.')
            parent = '.'.join(parts[:-1])
            leaf = parts[-1]
            if leaf.isdigit():
                bn_name = f"{parent}.{int(leaf)+1}" if parent else str(int(leaf)+1)
                bn_mod = dict(model.named_modules()).get(bn_name, None)
                if isinstance(bn_mod, nn.BatchNorm2d):
                    dep[n]["siblings"].append(bn_name)

            if i == len(convs) - 1:
                linear_names = [nnm for nnm, mm in model.named_modules() if isinstance(mm, nn.Linear)]
                if linear_names:
                    dep[n]["next"] += [linear_names[0]]
        return dep

    def apply_masks(self, model, masks: Dict[str, torch.Tensor]):
        new_model = copy.deepcopy(model)
        for layer_name, keep_mask in masks.items():
            if layer_name not in self.dependencies: continue
            idx = torch.where(keep_mask)[0]
            if idx.numel() == 0: idx = torch.tensor([0], device=keep_mask.device)
            orig_out = self.dependencies[layer_name]["out_channels"]
            
            self._shrink(new_model, layer_name, idx, dim=0)
            for sib in self.dependencies[layer_name]["siblings"]:
                self._shrink(new_model, sib, idx, dim=0)
            for nxt in self.dependencies[layer_name]["next"]:
                self._shrink(new_model, nxt, idx, dim=1, source_channels=orig_out)
        return new_model

    def _shrink(self, model, name, idx, dim, source_channels=None):
        parts = name.split('.')
        curr = model
        for p in parts[:-1]: curr = getattr(curr, p)
        module = getattr(curr, parts[-1])

        if isinstance(module, nn.Conv2d):
            if dim == 0:
                module.weight = nn.Parameter(module.weight.data[idx])
                module.out_channels = int(len(idx))
                if module.bias is not None:
                    module.bias = nn.Parameter(module.bias.data[idx])
            else:
                module.weight = nn.Parameter(module.weight.data[:, idx])
                module.in_channels = int(len(idx))

        elif isinstance(module, nn.BatchNorm2d):
            module.weight = nn.Parameter(module.weight.data[idx])
            module.bias = nn.Parameter(module.bias.data[idx])
            module.running_mean = module.running_mean[idx]
            module.running_var = module.running_var[idx]
            module.num_features = int(len(idx))

        elif isinstance(module, nn.Linear):
            if dim == 1:
                if source_channels is None or source_channels == 0:
                    new_idx = idx
                else:
                    scale = module.in_features // source_channels
                    scale = max(1, scale)
                    new_idx = torch.cat([idx * scale + s for s in range(scale)]).sort()[0]
                module.weight = nn.Parameter(module.weight.data[:, new_idx])
                module.in_features = int(len(new_idx))


class PyTorchAdapter(FrameworkAdapter):
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_model(self, model_type: str):
        if model_type == 'vgg16':
            m = models.vgg16_bn(num_classes=10)
            m.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            m.classifier = nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 10)
            )
            return m.to(self.device)
        elif model_type in ('resnet18', 'resnet'):
            m = models.resnet18(num_classes=10)
            m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            m.maxpool = nn.Identity()
            m.fc = nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 10)
            )
            return m.to(self.device)
        raise ValueError(f"Unsupported model_type: {model_type}")

    @timer
    def train(self, model, loader, epochs, name, val_loader=None, plot=True):
        from tqdm import tqdm
        lr = float(self.config.get('lr', 3e-4))
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for e in range(epochs):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            
            pbar = tqdm(loader, desc=f"[{name}] Epoch {e+1}/{epochs}")
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                
                total_loss += float(loss.item()) * y.size(0)
                correct += int((out.argmax(1) == y).sum().item())
                total += int(y.size(0))
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})

            avg_loss, avg_acc = total_loss / max(total, 1), 100.0 * correct / max(total, 1)
            hist['train_loss'].append(avg_loss)
            hist['train_acc'].append(avg_acc)

            if val_loader:
                v_loss, v_acc = self._loss_acc(model, val_loader)
                hist['val_loss'].append(v_loss)
                hist['val_acc'].append(v_acc)
                print(f"📊 Validation: Loss {v_loss:.4f}, Acc {v_acc:.2f}%")
        
        if plot:
            try:
                from ..visualization.stakeholder import plot_training_history
                plot_training_history(hist, title=name)
            except Exception as e:
                print(f"⚠️ Could not plot history: {e}")
                
        return hist

    def _loss_acc(self, model, loader):
        model.eval()
        crit = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = model(x)
                loss = crit(out, y)
                total_loss += float(loss.item()) * y.size(0)
                correct += int((out.argmax(1) == y).sum().item())
                total += int(y.size(0))
        return total_loss / max(total, 1), 100.0 * correct / max(total, 1)

    def evaluate(self, model, loader):
        _, acc = self._loss_acc(model, loader)
        return acc

    def get_viz_data(self, model, loader, num_layers=3):
        model.eval(); acts = {}; convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        hooks = [layer.register_forward_hook(lambda m, i, o, n=f"l{idx}": acts.update({n: o.detach()}))
                 for idx, layer in enumerate(convs[:num_layers])]
        x, _ = next(iter(loader)); x = x[:1].to(self.device)
        with torch.no_grad(): model(x)
        for h in hooks: h.remove()
        return {
            "activations": [v[0].cpu().numpy() for v in acts.values()],
            "filters": convs[0].weight.data.cpu().numpy()
        }

    def get_stats(self, model):
        m_cp = copy.deepcopy(model).cpu().eval()
        try:
            from thop import profile
            f, p = profile(m_cp, inputs=(torch.randn(1, 3, 32, 32),), verbose=False)
        except (ImportError, NameError):
            print("⚠️ thop not found. Returning (0.0, 0.0)")
            return 0.0, 0.0
        return float(f), float(p)

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)

    def load_checkpoint(self, model, path):
        model.load_state_dict(torch.load(path, map_location=self.device))

    def _activation_or_taylor_scores(self, model, loader, method: str) -> Dict[str, np.ndarray]:
        """Captures activations/gradients using hooks to calculate APoZ, Taylor, or mean activation scores."""
        model.eval()
        scores = {}
        hooks = []
        
        # Identify target layers
        convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        
        def get_hook(name):
            def hook(module, grad_input, grad_output):
                # Extract activation (output) or gradient (grad_output[0])
                if method == "taylor":
                    # Taylor = Abs(Activation * Gradient)
                    act = module.saved_act
                    grad = grad_output[0]
                    # Average over Batch, H, W
                    val = torch.abs(act * grad).mean(dim=(0, 2, 3)).detach().cpu().numpy()
                elif method == "apoz":
                    # APoZ = Mean percentage of zero activations
                    val = (grad_output.detach() <= 0).float().mean(dim=(0, 2, 3)).cpu().numpy()
                else: # mean_abs_act
                    val = torch.abs(grad_output.detach()).mean(dim=(0, 2, 3)).cpu().numpy()
                
                if name not in scores: scores[name] = []
                scores[name].append(val)
            return hook

        def forward_hook(module, input, output):
            module.saved_act = output.detach()

        # Register hooks
        for name, layer in convs:
            if method == "taylor":
                hooks.append(layer.register_forward_hook(forward_hook))
                hooks.append(layer.register_full_backward_hook(get_hook(name)))
            else:
                hooks.append(layer.register_forward_hook(lambda m, i, o, n=name: get_hook(n)(m, None, o)))

        # Run data through model
        criterion = nn.CrossEntropyLoss()
        for i, (x, y) in enumerate(loader):
            if i >= 5: break # Only use 5 batches for speed
            x, y = x.to(self.device), y.to(self.device)
            if method == "taylor":
                model.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
            else:
                with torch.no_grad():
                    model(x)

        # Cleanup hooks
        for h in hooks: h.remove()
        
        # Aggregate results (Mean across batches)
        return {n: np.mean(v, axis=0) for n, v in scores.items()}

    def get_score_map(self, model, loader, method: str) -> Dict[str, np.ndarray]:
        method = method.lower().strip()
        if method in ("taylor", "mean_abs_act", "apoz", "variance_act"):
            return self._activation_or_taylor_scores(model, loader, method)
            
        convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        score_map = {}
        for name, layer in convs:
            s = call_score_fn(method, {
                "layer_name": name,
                "layer": layer,
                "model": model,
                "loader": loader,
                "device": self.device
            })
            if s is None:
                raise ValueError(f"Method {method} returned None for layer {name}")
            if isinstance(s, torch.Tensor):
                s = s.detach().cpu().numpy()
            score_map[name] = np.asarray(s).reshape(-1)
        return score_map

    def apply_surgery(self, model, masks: Dict[str, np.ndarray]):
        convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        torch_masks = {
            n: torch.as_tensor(m, device=self.device, dtype=torch.bool)
            for n, m in masks.items() if n in dict(convs)
        }
        try:
            return TorchStructuralPruner(model).apply_masks(model, torch_masks)
        except Exception as e:
            raise SurgeryError(f"PyTorch structural surgery failed: {e}")
