import os
import copy
import numpy as np
from typing import Dict, Any, Callable, Tuple, List, Set, Optional
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
    # Framework-specific imports are wrapped in try-except to maintain
    # the library's ability to run (or be documented) without all backends.
    pass

class TorchStructuralPruner:
    """Enhanced structural surgeon with ResNet support via cluster tracing.

    This class handles the complexity of pruning PyTorch models where layers
    have interdependencies (e.g., residual connections, batch normalization).
    It ensures that pruning a layer also correctly updates all dependent layers.

    Attributes:
        dependencies (Dict[str, dict]): A mapping of layer names to their 
            dependency information and cluster assignments.
    """
    def __init__(self, model: nn.Module):
        """Initializes the pruner by tracing the model's structure.

        Args:
            model (nn.Module): The PyTorch model to be analyzed for pruning.
        """
        self.dependencies = self._trace(model)

    def _trace(self, model: nn.Module) -> Dict[str, dict]:
        """Traces dependencies using torch.fx for accurate graph connectivity."""
        import torch.fx
        try:
            # symbolic_trace works for most standard models (VGG, ResNet, etc.)
            traced = torch.fx.symbolic_trace(model)
            graph = traced.graph
        except Exception:
            # Fallback to a simplified sequential trace if FX fails
            return self._simple_trace(model)

        dep = {}
        convs = {n: m for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}
        node_to_name = {node: node.target for node in graph.nodes if node.op == 'call_module'}
        
        for node, name in node_to_name.items():
            if name in convs:
                dep[name] = {
                    "siblings": [],
                    "next": [],
                    "out_channels": convs[name].out_channels,
                    "cluster": None
                }
                
                # 1. Identify Batchnorm siblings via naming and graph
                parts = name.split('.')
                parent = '.'.join(parts[:-1])
                leaf = parts[-1]
                if leaf.startswith('conv'):
                    bn_name = f"{parent}.{leaf.replace('conv', 'bn')}" if parent else leaf.replace('conv', 'bn')
                    if dict(model.named_modules()).get(bn_name): dep[name]["siblings"].append(bn_name)
                elif leaf.isdigit():
                    bn_name = f"{parent}.{int(leaf)+1}" if parent else str(int(leaf)+1)
                    if isinstance(dict(model.named_modules()).get(bn_name), nn.BatchNorm2d):
                        dep[name]["siblings"].append(bn_name)

                # 2. Find next prunable layers (Conv/Linear) by traversing users
                queue = list(node.users.keys())
                visited = set()
                while queue:
                    curr = queue.pop(0)
                    if curr in visited: continue
                    visited.add(curr)
                    
                    if curr.op == 'call_module':
                        t_name = curr.target
                        t_mod = dict(model.named_modules()).get(t_name)
                        if isinstance(t_mod, (nn.Conv2d, nn.Linear)):
                            dep[name]["next"].append(t_name)
                        else:
                            queue.extend(curr.users.keys())
                    elif curr.op in ('call_function', 'call_method'):
                        queue.extend(curr.users.keys())

        # 3. Identify Clusters (ResNet Add nodes)
        cluster_id = 0
        for node in graph.nodes:
            if node.op == 'call_function' and any(x in str(node.target) for x in ('add', 'iadd')):
                # All convolutional producers of inputs to this Add must be in the same cluster
                members = []
                for in_node in node.all_input_nodes:
                    # Trace back to the nearest conv
                    q = [in_node]
                    v = set()
                    while q:
                        n_ = q.pop(0)
                        if n_ in v: continue
                        v.add(n_)
                        if n_.op == 'call_module' and n_.target in dep:
                            members.append(n_.target)
                        elif hasattr(n_, "all_input_nodes"):
                            q.extend(n_.all_input_nodes)
                
                if len(members) > 1:
                    for m in members: dep[m]["cluster"] = cluster_id
                    cluster_id += 1
        return dep

    def _simple_trace(self, model: nn.Module) -> Dict[str, dict]:
        """Simple sequential fallback trace."""
        dep = {}
        convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        names = [n for n, _ in convs]
        for i, (n, m) in enumerate(convs):
            dep[n] = {"siblings": [], "next": [names[i+1]] if i+1 < len(names) else [],
                      "out_channels": m.out_channels, "cluster": None}
        return dep

    def apply_masks(self, model: nn.Module, masks: Dict[str, np.ndarray]) -> nn.Module:
        """Applies pruning masks to the model, harmonizing across clusters.

        Args:
            model (nn.Module): The original model.
            masks (Dict[str, np.ndarray]): Dictionary mapping layer names to 
                binary masks (1 = keep, 0 = prune).

        Returns:
            nn.Module: A new, structurally pruned model (smaller tensors).
        """
        new_model = copy.deepcopy(model)
        
        # 1. Harmonize masks for clusters
        # If any layer in a cluster is pruned, all other layers in that 
        # cluster must use the same mask to maintain compatibility.
        processed_clusters = set()
        harmonized_masks = copy.deepcopy(masks)
        
        for name, mask in masks.items():
            if name not in self.dependencies: continue
            c_id = self.dependencies[name]["cluster"]
            if c_id is not None and c_id not in processed_clusters:
                # Find all members of this cluster and enforce the same mask
                cluster_members = [n for n, d in self.dependencies.items() if d["cluster"] == c_id]
                anchor_mask = harmonized_masks[name]
                for member in cluster_members:
                    harmonized_masks[member] = anchor_mask
                processed_clusters.add(c_id)

        # 2. Apply Surgery: Recursively shrink layer parameters
        processed_in_shrinks = set()
        for layer_name, keep_mask in harmonized_masks.items():
            if layer_name not in self.dependencies: continue
            
            # Find indices of channels to keep
            idx = torch.where(torch.as_tensor(keep_mask))[0]
            # Safety: don't prune everything, keep at least one channel
            if idx.numel() == 0: 
                idx = torch.tensor([0], device=new_model.parameters().__next__().device)
            
            orig_out = self.dependencies[layer_name]["out_channels"]
            
            # Shrink the output dimension of the current layer
            self._shrink(new_model, layer_name, idx, dim=0)
            
            # Shrink coupled layers (like BatchNorm)
            for sib in self.dependencies[layer_name]["siblings"]:
                self._shrink(new_model, sib, idx, dim=0)
            
            # Shrink the input dimension of subsequent layers
            for nxt in self.dependencies[layer_name]["next"]:
                # Avoid double-shrinking the same input if multiple cluster 
                # members feed into it (e.g., after an Add node).
                shrink_key = f"{nxt}_in"
                if shrink_key not in processed_in_shrinks:
                    self._shrink(new_model, nxt, idx, dim=1, source_channels=orig_out)
                    processed_in_shrinks.add(shrink_key)
        
        return new_model

    def _shrink(self, model: nn.Module, name: str, idx: torch.Tensor, dim: int, 
                source_channels: Optional[int] = None) -> None:
        """Low-level tensor shrinking operation.

        Physically removes weights/biases corresponding to pruned indices.

        Args:
            model (nn.Module): The model being modified.
            name (str): Full path to the module (e.g., 'features.0').
            idx (torch.Tensor): Indices of channels to keep.
            dim (int): Dimension to shrink (0 for output, 1 for input).
            source_channels (Optional[int]): Original number of channels in the source
                layer (used for handling flattened Linear layers after Convs).
        """
        parts = name.split('.')
        curr = model
        # Traverse the model hierarchy to find the target module
        for p in parts[:-1]: 
            curr = getattr(curr, p)
        module = getattr(curr, parts[-1])

        if isinstance(module, nn.Conv2d):
            # print(f"DEBUG: Shrinking {name} dim={dim} idx_size={len(idx)} weight_shape={module.weight.shape}")
            if dim == 0:
                # Shrinking output channels
                module.weight = nn.Parameter(module.weight.data[idx])
                module.out_channels = int(len(idx))
                if module.bias is not None:
                    module.bias = nn.Parameter(module.bias.data[idx])
            else:
                # Shrinking input channels
                if module.groups == 1:
                    module.weight = nn.Parameter(module.weight.data[:, idx])
                    module.in_channels = int(len(idx))
                else:
                    # Depthwise or Grouped convolution handling
                    module.weight = nn.Parameter(module.weight.data[idx])
                    module.in_channels = int(len(idx))
                    module.groups = int(len(idx))

        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm layers are always shrunk along the feature dimension
            module.weight = nn.Parameter(module.weight.data[idx])
            module.bias = nn.Parameter(module.bias.data[idx])
            module.running_mean = module.running_mean[idx]
            module.running_var = module.running_var[idx]
            module.num_features = int(len(idx))

        elif isinstance(module, nn.Linear):
            if dim == 1:
                # Linear layers after convolutions need special handling if spatial
                # info was flattened (e.g., 512 channels -> 512*7*7 units).
                in_f = module.in_features
                if source_channels and in_f != source_channels:
                    # Map pruned channel indices to the expanded flattened indices
                    scale = in_f // source_channels
                    new_idx = torch.cat([idx * scale + s for s in range(scale)]).sort()[0]
                else:
                    new_idx = idx
                module.weight = nn.Parameter(module.weight.data[:, new_idx])
                module.in_features = int(len(new_idx))

class PyTorchAdapter(FrameworkAdapter):
    """Bridge between the core pruner and the PyTorch framework.

    Implements the standard interface defined by FrameworkAdapter to allow
    the pruner to interact with PyTorch models, data loaders, and training loops.
    """
    def __init__(self, config: dict = None):
        """Initializes the adapter.

        Args:
            config (dict, optional): Configuration parameters. Defaults to None.
        """
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_model(self, model_type: str, input_shape: Tuple[int, int, int] = None, 
                  num_classes: int = None) -> nn.Module:
        """Factory method to create standard models.

        Args:
            model_type (str): Type of model (e.g., 'vgg16', 'resnet18').
            input_shape (Tuple[int, int, int], optional): (C, H, W). Defaults to None.
            num_classes (int, optional): Number of output classes. Defaults to None.

        Returns:
            nn.Module: Initialized PyTorch model on the appropriate device.
        """
        from torchvision import models
        input_shape = input_shape or self.config.get('input_shape', (3, 32, 32))
        num_classes = num_classes or self.config.get('num_classes', 10)
        channels, height, width = input_shape

        if model_type == 'vgg16':
            m = models.vgg16_bn(num_classes=num_classes)
            # Adjust input channels if not 3
            if channels != 3:
                m.features[0] = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
            # Standardize output for small images (CIFAR-style)
            m.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            m.classifier = nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, num_classes)
            )
            return m.to(self.device)
        elif model_type in ('resnet18', 'resnet'):
            m = models.resnet18(num_classes=num_classes)
            # Adjust first conv for 32x32 images
            m.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if height < 64:
                m.maxpool = nn.Identity()
            m.fc = nn.Linear(512, num_classes)
            return m.to(self.device)
        raise ValueError(f"Unsupported model_type: {model_type}")

    @timer
    def train(self, model: nn.Module, loader: Any, epochs: int, name: str, 
              val_loader: Any = None, plot: bool = True) -> dict:
        """Standard PyTorch training loop.

        Args:
            model (nn.Module): The model to train.
            loader (Any): Training data loader.
            epochs (int): Number of epochs.
            name (str): Label for the training run.
            val_loader (Any, optional): Validation data loader. Defaults to None.
            plot (bool, optional): Whether to plot history. Defaults to True.

        Returns:
            dict: Training history (loss and accuracy).
        """
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
            except Exception: pass
        return hist

    def _loss_acc(self, model: nn.Module, loader: Any) -> Tuple[float, float]:
        """Calculates loss and accuracy on a dataset.

        Args:
            model (nn.Module): The model to evaluate.
            loader (Any): Data loader.

        Returns:
            Tuple[float, float]: (average_loss, accuracy_percentage).
        """
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

    def evaluate(self, model: nn.Module, loader: Any) -> float:
        """Evaluates model accuracy.

        Args:
            model (nn.Module): The model to evaluate.
            loader (Any): Data loader.

        Returns:
            float: Accuracy percentage.
        """
        _, acc = self._loss_acc(model, loader)
        return acc

    def get_viz_data(self, model: nn.Module, loader: Any, num_layers: int = 3) -> dict:
        """Extracts activation and filter data for visualization.

        Args:
            model (nn.Module): The model.
            loader (Any): Data loader for a sample input.
            num_layers (int, optional): Number of layers to extract. Defaults to 3.

        Returns:
            dict: Activation maps and filter weights.
        """
        model.eval()
        acts = {}
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        
        # Register temporary hooks to capture intermediate activations
        hooks = [layer.register_forward_hook(lambda m, i, o, n=f"l{idx}": acts.update({n: o.detach()}))
                 for idx, layer in enumerate(convs[:num_layers])]
        
        it = iter(loader)
        x, _ = next(it)
        x = x[:1].to(self.device)
        with torch.no_grad(): 
            model(x)
            
        # Cleanup hooks
        for h in hooks: 
            h.remove()
            
        return {
            "activations": [v[0].cpu().numpy() for v in acts.values()], 
            "filters": convs[0].weight.data.cpu().numpy()
        }

    def get_stats(self, model: nn.Module, loader: Any = None) -> Tuple[float, float]:
        """Calculates FLOPs and Parameter count.

        Args:
            model (nn.Module): The model to analyze.
            loader (Any, optional): Data loader to infer input shape. Defaults to None.

        Returns:
            Tuple[float, float]: (FLOPs, total_parameters).
        """
        m_cp = copy.deepcopy(model).cpu().eval()
        if loader:
            x, _ = next(iter(loader))
            in_shape = tuple(x.shape[1:])
        else:
            in_shape = self.config.get('input_shape', (3, 32, 32))
        try:
            from thop import profile
            # Using 'thop' (Torch-Handling-Of-Params) for robust FLOP estimation
            f, p = profile(m_cp, inputs=(torch.randn(1, *in_shape),), verbose=False)
        except Exception: 
            return 0.0, 0.0
        return float(f), float(p)

    def save_checkpoint(self, model: nn.Module, path: str):
        """Saves model state dict."""
        torch.save(model.state_dict(), path)
        
    def load_checkpoint(self, model: nn.Module, path: str):
        """Loads model state dict."""
        model.load_state_dict(torch.load(path, map_location=self.device))

    def _activation_or_taylor_scores(self, model: nn.Module, loader: Any, method: str) -> Dict[str, np.ndarray]:
        """Calculates channel scores using dynamic activation-based methods.

        Supports: 'taylor' (first-order), 'mean_abs_act', 'apoz', 'variance_act'.

        Args:
            model (nn.Module): The model.
            loader (Any): Calibration data.
            method (str): Scoring method.

        Returns:
            Dict[str, np.ndarray]: Mapping of layer names to channel importance scores.
        """
        model.eval()
        scores = {}
        hooks = []
        convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        
        def get_hook(name):
            def hook(module, grad_input, grad_output):
                o = grad_output[0] if isinstance(grad_output, tuple) else grad_output
                if method == "taylor":
                    act = module.saved_act
                    # Taylor score: |act * grad|
                    val = torch.abs(act * o).mean(dim=(0, 2, 3)).detach().cpu().numpy()
                elif method == "apoz": 
                    # Average Percentage of Zeros
                    val = (o.detach() <= 0).float().mean(dim=(0, 2, 3)).cpu().numpy()
                else: 
                    # Mean Absolute Activation
                    val = torch.abs(o.detach()).mean(dim=(0, 2, 3)).cpu().numpy()
                
                if name not in scores: 
                    scores[name] = []
                scores[name].append(val)
            return hook
            
        def forward_hook(module, input, output): 
            # Cache activations for Taylor expansion
            module.saved_act = output.detach()
            
        for name, layer in convs:
            if method == "taylor":
                hooks.append(layer.register_forward_hook(forward_hook))
                hooks.append(layer.register_full_backward_hook(get_hook(name)))
            else: 
                hooks.append(layer.register_forward_hook(lambda m, i, o, n=name: get_hook(n)(m, None, o)))
        
        crit = nn.CrossEntropyLoss()
        # Run calibration batches
        for i, (x, y) in enumerate(loader):
            if i >= 5: break # Standard limit for calibration
            x, y = x.to(self.device), y.to(self.device)
            if method == "taylor":
                model.zero_grad()
                out = model(x)
                loss = crit(out, y)
                loss.backward()
            else:
                with torch.no_grad(): 
                    model(x)
                    
        for h in hooks: 
            h.remove()
        return {n: np.mean(v, axis=0) for n, v in scores.items()}

    def get_score_map(self, model: nn.Module, loader: Any, method: str) -> Dict[str, np.ndarray]:
        """Generates a complete score map for all prunable layers.

        Args:
            model (nn.Module): The model.
            loader (Any): Calibration data.
            method (str): The pruning criterion (e.g., 'l1_norm', 'taylor').

        Returns:
            Dict[str, np.ndarray]: Channel importance scores.
        """
        method = method.lower().strip()
        # Handle specialized activation-based methods
        if method in ("taylor", "mean_abs_act", "apoz", "variance_act"): 
            return self._activation_or_taylor_scores(model, loader, method)
            
        # Handle registered weight-based or hybrid methods
        convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        score_map = {}
        for name, layer in convs:
            s = call_score_fn(method, "torch", {
                "layer_name": name, "layer": layer, "model": model, 
                "loader": loader, "device": self.device
            })
            if s is None: 
                raise ValueError(f"Method {method} returned None for layer {name}")
            if isinstance(s, torch.Tensor): 
                s = s.detach().cpu().numpy()
            score_map[name] = np.asarray(s).reshape(-1)
        return score_map

    def apply_surgery(self, model: nn.Module, masks: Dict[str, np.ndarray]) -> nn.Module:
        """Performs the structural pruning.

        Args:
            model (nn.Module): Original model.
            masks (Dict[str, np.ndarray]): Pruning masks.

        Returns:
            nn.Module: The new, smaller model.
        """
        try: 
            return TorchStructuralPruner(model).apply_masks(model, masks)
        except Exception as e: 
            raise SurgeryError(f"PyTorch structural surgery failed: {e}")
