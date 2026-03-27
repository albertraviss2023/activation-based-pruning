import os
import copy
import numpy as np
from typing import Dict, Any, Callable, Tuple, List, Set, Optional
from pathlib import Path
from datetime import datetime
from ..core.adapter import FrameworkAdapter
from ..core.decorators import timer
from ..core.exceptions import SurgeryError
from ..pruner.registry import get_method, call_score_fn
from ..pruner.custom_method_tools import CustomMethodTools

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.fx
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

    def _trace(self, model: nn.Module) -> Dict[str, Any]:
        """Traces dependencies using torch.fx for accurate graph connectivity.
        
        Returns:
            Dict[str, Any]: Standardized graph format:
                {
                    "nodes": { name: { type, inputs, outputs, cluster_id } },
                    "clusters": { id: [names] }
                }
        """
        try:
            traced = torch.fx.symbolic_trace(model)
            graph = traced.graph
        except Exception:
            # Fallback to a simplified sequential trace
            return self._simple_trace_standardized(model)

        nodes_info = {}
        # We now consider both Conv2d and Linear as prunable.
        # However, we often want to avoid pruning the final classification layer's output.
        prunables = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))}
        node_to_name = {node: node.target for node in graph.nodes if node.op == 'call_module'}
        
        # Identify the last linear/conv layer (likely the output head)
        last_prunable_name = None
        for node in reversed(list(graph.nodes)):
            if node.op == 'call_module' and node.target in prunables:
                last_prunable_name = node.target
                break

        def nearest_prunable_inputs(node) -> List[str]:
            """Find nearest upstream prunable producers through non-prunable ops."""
            parents: Set[str] = set()
            queue = list(getattr(node, "all_input_nodes", []))
            visited = set()
            while queue:
                n_ = queue.pop(0)
                if n_ in visited:
                    continue
                visited.add(n_)
                if n_.op == 'call_module' and n_.target in prunables:
                    parents.add(n_.target)
                    continue
                if hasattr(n_, "all_input_nodes"):
                    queue.extend(n_.all_input_nodes)
            return sorted(parents)

        def nearest_prunable_outputs(node) -> List[str]:
            """Find nearest downstream prunable consumers through non-prunable ops."""
            children: Set[str] = set()
            queue = list(getattr(node, "users", {}).keys())
            visited = set()
            while queue:
                n_ = queue.pop(0)
                if n_ in visited:
                    continue
                visited.add(n_)
                if n_.op == 'call_module' and n_.target in prunables:
                    children.add(n_.target)
                    continue
                if hasattr(n_, "users"):
                    queue.extend(list(n_.users.keys()))
            return sorted(children)

        # 1. Identify prunable nodes and their siblings (BN), with connectivity.
        for node, name in node_to_name.items():
            if name in prunables:
                nodes_info[name] = {
                    "type": "conv2d" if isinstance(prunables[name], nn.Conv2d) else "linear",
                    "inputs": nearest_prunable_inputs(node),
                    "outputs": nearest_prunable_outputs(node),
                    "siblings": [],
                    "cluster": None,
                    "is_output_head": (name == last_prunable_name)
                }
                # Identify Batchnorm siblings via graph connectivity
                for user_node in getattr(node, "users", {}):
                    if user_node.op == 'call_module':
                        user_mod = dict(model.named_modules()).get(user_node.target)
                        if isinstance(user_mod, nn.BatchNorm2d):
                            nodes_info[name]["siblings"].append(user_node.target)

        # 2. Identify raw residual clusters from Add nodes.
        # We later merge overlaps into disjoint connected components so
        # harmonization is deterministic.
        raw_clusters = []
        for node in graph.nodes:
            # Handle Addition (Residual) - these require identical masks
            if node.op == 'call_function' and any(x in str(node.target) for x in ('add', 'iadd')):
                members = []
                q = [node]
                v = set()
                while q:
                    n_ = q.pop(0)
                    if n_ in v: continue
                    v.add(n_)
                    if n_.op == 'call_module' and n_.target in nodes_info:
                        members.append(n_.target)
                    elif hasattr(n_, "all_input_nodes"):
                        q.extend(n_.all_input_nodes)
                
                if len(members) > 1:
                    raw_clusters.append(set(members))
            
            # Handle Concatenation (DenseNet)
            elif node.op == 'call_function' and 'cat' in str(node.target):
                # For cat, we don't necessarily need clusters (identical masks), 
                # but we need to track that multiple layers contribute to the 
                # input of the next layer.
                pass
                    
        # Merge overlapping raw clusters into disjoint connected components.
        merged = []
        for c in raw_clusters:
            found = False
            for m in merged:
                if not c.isdisjoint(m):
                    m.update(c)
                    found = True
                    break
            if not found:
                merged.append(set(c))

        clusters = {}
        for c_id, members in enumerate(merged):
            clusters[c_id] = sorted(members)
            for m in members:
                if m in nodes_info:
                    nodes_info[m]["cluster"] = c_id

        return {"nodes": nodes_info, "clusters": clusters}

    def _simple_trace_standardized(self, model: nn.Module) -> Dict[str, Any]:
        """Simple sequential fallback trace in standardized format."""
        nodes_info = {}
        convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        for i, (n, m) in enumerate(convs):
            nodes_info[n] = {
                "type": "conv2d",
                "inputs": [convs[i-1][0]] if i > 0 else [],
                "outputs": [convs[i+1][0]] if i+1 < len(convs) else [],
                "siblings": [],
                "cluster": None
            }
        return {"nodes": nodes_info, "clusters": {}}

    def apply_masks(self, model: nn.Module, masks: Dict[str, np.ndarray]) -> nn.Module:
        """Applies pruning masks to the model, harmonizing across clusters and ensuring
        consistent shape propagation across skip connections.
        """
        new_model = copy.deepcopy(model)
        device = next(model.parameters()).device
        
        # 1. Harmonize masks for clusters (Residual Add nodes)
        harmonized_masks = copy.deepcopy(masks)
        for c_id, members in self.dependencies["clusters"].items():
            anchor_name = next((m for m in members if m in harmonized_masks), None)
            if anchor_name:
                anchor_mask = torch.as_tensor(harmonized_masks[anchor_name], device='cpu')
                for member in members:
                    harmonized_masks[member] = anchor_mask.numpy()

        # 2. Setup structural tracing for physical surgery plan
        plan = {}
        def set_plan(name, idx, dim, src_ch=None):
            if name not in plan: plan[name] = {}
            # Use 'out' and 'in' for clarity
            key = "dim0" if dim == 0 else "dim1"
            plan[name][key] = idx
            if src_ch: plan[name]["src_ch"] = src_ch

        traced = torch.fx.symbolic_trace(model)
        
        # Run dummy pass to record shapes for 'cat' and 'Linear' expansion handling
        def shape_hook(module, input, output):
            module.recorded_shape = output.shape
        hooks = []
        for n, m in model.named_modules():
            hooks.append(m.register_forward_hook(shape_hook))
        
        try:
            first_conv = next((m for m in model.modules() if isinstance(m, nn.Conv2d)), None)
            in_ch = int(first_conv.in_channels) if first_conv else 3
            for size in (32, 64, 128, 224):
                try:
                    dummy_in = torch.randn(1, in_ch, size, size, device=device)
                    model(dummy_in)
                    break
                except Exception: continue
        except Exception: pass
        for h in hooks: h.remove()

        # node -> torch.Tensor (indices to keep for the TENSOR produced by this node)
        node_masks = {} 
        
        for node in traced.graph.nodes:
            if node.op == 'placeholder':
                continue
            
            elif node.op == 'call_module':
                m_name = node.target
                m = dict(model.named_modules()).get(m_name)
                in_idx_consistent = None
                
                # A. Input pruning for this module
                # Check ALL inputs. If multiple inputs (like in add), they must ALL have masks
                # and those masks must be identical for us to prune the INPUT dimension of this layer.
                input_masks = [node_masks[in_n] for in_n in node.all_input_nodes if in_n in node_masks]
                
                if input_masks:
                    # Verify consistency - for 'add' nodes, all input masks MUST be the same
                    # If we only have one mask, it's straightforward.
                    in_idx = input_masks[0]
                    is_consistent = all(len(m) == len(in_idx) and torch.equal(m, in_idx) for m in input_masks)
                    
                    if is_consistent:
                        in_idx_consistent = in_idx
                        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                            dim = 1 if not isinstance(m, nn.BatchNorm2d) else 0
                            
                            # Handle Linear expansion for flattened inputs
                            if isinstance(m, nn.Linear) and m.in_features > len(in_idx):
                                def get_orig_channels(n_):
                                    if n_.op == 'call_module':
                                        pm = dict(model.named_modules()).get(n_.target)
                                        return getattr(pm, 'recorded_shape', [0,0,0,0])[1]
                                    elif n_.all_input_nodes: return get_orig_channels(n_.all_input_nodes[0])
                                    return 0
                                
                                orig_c = get_orig_channels(node.all_input_nodes[0])
                                if orig_c > 0 and m.in_features % orig_c == 0:
                                    scale = m.in_features // orig_c
                                    in_idx = torch.cat([in_idx * scale + s for s in range(scale)]).sort()[0]
                            
                            set_plan(m_name, in_idx, dim)

                # B. Output pruning for this module
                if m_name in harmonized_masks:
                    m_idx = torch.where(torch.as_tensor(harmonized_masks[m_name]))[0]
                    if m_idx.numel() == 0: m_idx = torch.tensor([0], device=device)
                    # Depthwise convs in MobileNet-like blocks require input/output/channel-group
                    # consistency; enforce it at planning time to avoid invalid grouped kernels.
                    is_depthwise = (
                        isinstance(m, nn.Conv2d)
                        and m.groups == m.in_channels
                        and m.out_channels == m.in_channels
                        and m.groups > 1
                    )
                    if is_depthwise and in_idx_consistent is not None and in_idx_consistent.numel() > 0:
                        # For depthwise conv, connectivity is governed by producer output channels.
                        # Follow input mask to keep in/out/groups aligned and avoid runtime mismatch.
                        m_idx = in_idx_consistent
                    set_plan(m_name, m_idx, 0)
                    if is_depthwise:
                        set_plan(m_name, m_idx, 1)
                    for sib in self.dependencies["nodes"].get(m_name, {}).get("siblings", []):
                        set_plan(sib, m_idx, 0)
                    node_masks[node] = m_idx
                elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU, 
                                   nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
                                   nn.Dropout, nn.Dropout2d, nn.Identity, nn.BatchNorm2d)):
                    # Propagate mask through transparent layers
                    if node.all_input_nodes and node.all_input_nodes[0] in node_masks:
                        node_masks[node] = node_masks[node.all_input_nodes[0]]
            
            elif node.op == 'call_function':
                if 'cat' in str(node.target):
                    combined_idx = []
                    offset = 0
                    for in_n in node.args[0]:
                        def get_out_ch(n_):
                            if n_.op == 'call_module':
                                mod = dict(model.named_modules()).get(n_.target)
                                return getattr(mod, 'recorded_shape', [0,0,0,0])[1]
                            elif n_.all_input_nodes: return get_out_ch(n_.all_input_nodes[0])
                            return 0
                        
                        ch_count = get_out_ch(in_n)
                        if in_n in node_masks:
                            combined_idx.append(node_masks[in_n] + offset)
                        elif ch_count > 0:
                            combined_idx.append(torch.arange(ch_count, device=device) + offset)
                        offset += ch_count
                    if combined_idx:
                        node_masks[node] = torch.cat(combined_idx)
                
                elif any(x in str(node.target) for x in ('add', 'iadd')):
                    # Residual Add: Output mask only exists if ALL inputs are pruned identically
                    input_masks = [node_masks[in_n] for in_n in node.all_input_nodes if in_n in node_masks]
                    if len(input_masks) == len(node.all_input_nodes):
                        # Verify all paths are pruned the same way
                        m0 = input_masks[0]
                        if all(len(m) == len(m0) and torch.equal(m, m0) for m in input_masks):
                            node_masks[node] = m0
                else:
                    # Generic transparent function (e.g. relu, adaptive_avg_pool)
                    if node.all_input_nodes and node.all_input_nodes[0] in node_masks:
                        node_masks[node] = node_masks[node.all_input_nodes[0]]
            
            elif node.op == 'call_method':
                if node.target in ('view', 'flatten', 'reshape'):
                    in_n = node.all_input_nodes[0]
                    if in_n in node_masks:
                        in_idx = node_masks[in_n]
                        def get_producer_info(n_):
                            if n_.op == 'call_module':
                                mod = dict(model.named_modules()).get(n_.target)
                                if isinstance(mod, (nn.Conv2d, nn.BatchNorm2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AvgPool2d)):
                                    s = getattr(mod, 'recorded_shape', [0,0,0,0])
                                    return np.prod(s[1:]), s[1]
                            if n_.all_input_nodes: return get_producer_info(n_.all_input_nodes[0])
                            return 0, 0
                        total_f, channels = get_producer_info(in_n)
                        if channels > 0 and total_f > 0:
                            scale = total_f // channels
                            node_masks[node] = torch.cat([in_idx * scale + s for s in range(scale)]).sort()[0]
                elif node.all_input_nodes and node.all_input_nodes[0] in node_masks:
                    node_masks[node] = node_masks[node.all_input_nodes[0]]

        # 3. Execute Unified Plan
        for name, p in plan.items():
            self._execute_layer_surgery(new_model, name, p)
        
        return new_model

    def _execute_layer_surgery(self, model: nn.Module, name: str, p: Dict[str, Any]):
        """Applies all dimension shrinks for a single layer at once."""
        parts = name.split('.')
        curr = model
        for part in parts[:-1]: curr = getattr(curr, part)
        module = getattr(curr, parts[-1])
        
        dim0_idx = p.get("dim0")
        dim1_idx = p.get("dim1")
        src_ch = p.get("src_ch")

        if isinstance(module, nn.Conv2d):
            is_depthwise = (
                module.groups == module.in_channels
                and module.out_channels == module.in_channels
                and module.groups > 1
            )
            if is_depthwise:
                # Prefer producer-driven input keep set for depthwise layers.
                source_idx = dim1_idx if dim1_idx is not None else dim0_idx
                if source_idx is not None:
                    keep = source_idx.detach().cpu().numpy().astype(np.int64).reshape(-1)
                    keep = np.unique(keep)
                    keep = keep[(keep >= 0) & (keep < module.weight.data.shape[0])]
                    if keep.size == 0:
                        keep = np.array([0], dtype=np.int64)
                    keep_t = torch.as_tensor(keep, dtype=torch.long, device=module.weight.device)
                    module.weight = nn.Parameter(module.weight.data[keep_t])
                    if module.bias is not None:
                        module.bias = nn.Parameter(module.bias.data[keep_t])
                    k = int(len(keep_t))
                    module.out_channels = k
                    module.in_channels = k
                    module.groups = k
                return

            if dim0_idx is not None:
                module.weight = nn.Parameter(module.weight.data[dim0_idx])
                module.out_channels = len(dim0_idx)
                if module.bias is not None:
                    module.bias = nn.Parameter(module.bias.data[dim0_idx])
            if dim1_idx is not None:
                if module.groups == 1:
                    module.weight = nn.Parameter(module.weight.data[:, dim1_idx])
                    module.in_channels = len(dim1_idx)
                else:
                    # Depthwise handling (groups == in_channels == out_channels)
                    # For depthwise, we already shrunk dim0 above.
                    module.in_channels = len(dim1_idx)
                    module.groups = len(dim1_idx)

        elif isinstance(module, nn.BatchNorm2d):
            # For BN, dim0 and dim1 (if it came from a user path) refer to same dimension
            idx = dim0_idx if dim0_idx is not None else dim1_idx
            if idx is not None:
                module.weight = nn.Parameter(module.weight.data[idx])
                module.bias = nn.Parameter(module.bias.data[idx])
                module.running_mean = module.running_mean[idx]
                module.running_var = module.running_var[idx]
                module.num_features = len(idx)

        elif isinstance(module, nn.Linear):
            if dim0_idx is not None:
                module.weight = nn.Parameter(module.weight.data[dim0_idx])
                module.out_features = len(dim0_idx)
                if module.bias is not None:
                    module.bias = nn.Parameter(module.bias.data[dim0_idx])
            if dim1_idx is not None:
                in_f = module.in_features
                if src_ch and in_f != src_ch:
                    scale = in_f // src_ch
                    new_idx = torch.cat([dim1_idx * scale + s for s in range(scale)]).sort()[0]
                else:
                    new_idx = dim1_idx
                module.weight = nn.Parameter(module.weight.data[:, new_idx])
                module.in_features = len(new_idx)

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
            is_depthwise = (
                module.groups == module.in_channels
                and module.out_channels == module.in_channels
                and module.groups > 1
            )
            if is_depthwise:
                keep = idx.detach().cpu().numpy().astype(np.int64).reshape(-1)
                keep = np.unique(keep)
                keep = keep[(keep >= 0) & (keep < module.weight.data.shape[0])]
                if keep.size == 0:
                    keep = np.array([0], dtype=np.int64)
                keep_t = torch.as_tensor(keep, dtype=torch.long, device=module.weight.device)
                module.weight = nn.Parameter(module.weight.data[keep_t])
                if module.bias is not None:
                    module.bias = nn.Parameter(module.bias.data[keep_t])
                k = int(len(keep_t))
                module.out_channels = k
                module.in_channels = k
                module.groups = k
                return

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

    def _resolve_baseline_dir(self, model: nn.Module) -> Path:
        cfg = self.config or {}
        model_type = str(
            cfg.get("model_type")
            or cfg.get("model_name")
            or model.__class__.__name__.lower()
        ).lower().strip()
        dataset_key = str(
            cfg.get("dataset_key")
            or cfg.get("dataset")
            or os.environ.get("REDUCNN_DATASET_KEY")
            or "dataset"
        ).lower().strip()
        return Path("saved_models") / "baselines" / "pytorch" / dataset_key / model_type

    def _latest_baseline_ckpt(self, model: nn.Module) -> Optional[Path]:
        d = self._resolve_baseline_dir(model)
        if not d.exists():
            return None
        files = sorted(d.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None

    def _new_baseline_ckpt(self, model: nn.Module) -> Path:
        d = self._resolve_baseline_dir(model)
        d.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = d.name
        dataset_key = d.parent.name
        return d / f"{stamp}_pytorch_{model_type}_{dataset_key}_baseline.pth"

    @staticmethod
    def _is_baseline_name(name: str) -> bool:
        n = str(name or "").lower()
        return ("baseline" in n) or ("base" in n and "heal" not in n and "finetune" not in n)

    def _iter_prunable_layers(
        self, model: nn.Module, deps: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, nn.Module]]:
        """Returns prunable layers while avoiding over-aggressive head exclusion.

        We only skip the final output head when it is a trailing Linear layer and
        there are other prunable layers available. This keeps tiny/conv-only test
        models prunable and prevents empty score maps.
        """
        deps = deps or self.trace_graph(model)
        all_prunables: List[Tuple[str, nn.Module]] = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]
        if not all_prunables:
            return []

        skip_name = None
        if len(all_prunables) > 1:
            last_name, last_module = all_prunables[-1]
            is_head = bool(deps["nodes"].get(last_name, {}).get("is_output_head", False))
            if is_head and isinstance(last_module, nn.Linear):
                skip_name = last_name

        prunables = [(name, module) for name, module in all_prunables if name != skip_name]
        return prunables or all_prunables

    def get_global_activations(self, model: torch.nn.Module, loader: Any, num_batches: int = 1) -> Dict[str, np.ndarray]:
        """
        Collects average activation magnitudes for all prunable layers in a single pass.
        Useful for global network flow animations.
        """
        model.eval()
        acts = {}
        hooks = []
        
        def get_hook(name):
            def hook(m, i, o):
                if len(o.shape) == 4:
                    val = o.mean(dim=(2, 3)).detach().cpu().numpy()
                else:
                    val = o.detach().cpu().numpy()
                if name not in acts:
                    acts[name] = []
                acts[name].append(val)
            return hook
            
        prunables = {name: module for name, module in model.named_modules() 
                     if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))}
        
        for name, module in prunables.items():
            hooks.append(module.register_forward_hook(get_hook(name)))
            
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= num_batches: break
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                model(x.to(self.device))
                
        for h in hooks: 
            h.remove()
            
        # Average across the batch dimension to get a single magnitude per channel
        return {k: np.concatenate(v, axis=0).mean(axis=0) for k, v in acts.items()}

    def get_layer_activations(self, model: torch.nn.Module, loader: Any, layer_name: str, num_batches: int = 1) -> np.ndarray:
        """
        Collects raw activations for a specific layer.
        Returns array of shape [num_samples, channels]
        """
        model.eval()
        activations = []
        
        def hook(m, i, o):
            # Global average pool to get [batch, channels]
            if len(o.shape) == 4:
                # [batch, channels, h, w] -> [batch, channels]
                activations.append(o.mean(dim=(2, 3)).detach().cpu().numpy())
            else:
                activations.append(o.detach().cpu().numpy())
                
        # Find the layer
        target_layer = None
        for name, module in model.named_modules():
            if name == layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Layer {layer_name} not found in model.")
            
        handle = target_layer.register_forward_hook(hook)
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= num_batches: break
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                model(x.to(self.device))
                
        handle.remove()
        return np.concatenate(activations, axis=0)

    def get_layer_importance(self, model: torch.nn.Module, loader: Any, layer_name: str, method: str = 'l1_norm') -> np.ndarray:
        """
        Computes importance scores for a specific layer using the requested method.
        """
        from ..pruner.registry import CRITERIA_REGISTRY
        if method not in CRITERIA_REGISTRY:
            raise ValueError(f"Method {method} not registered.")
            
        score_map = CRITERIA_REGISTRY[method](self, model, loader)
        
        if layer_name in score_map:
            return score_map[layer_name]
            
        # Fallback: exact match or partial match for recursive modules
        for name, score in score_map.items():
            if name == layer_name or name.endswith(f".{layer_name}"):
                return score
                
        raise ValueError(f"No importance scores found for layer {layer_name} using {method}.")

    def get_model(self, model_type: str, input_shape: Tuple[int, int, int] = None, 
                  num_classes: int = None, pretrained: bool = False) -> nn.Module:
        """Factory method to create models. Supports torchvision models by name.

        Args:
            model_type (str): Type of model (e.g., 'vgg16', 'resnet18', 'mobilenet_v2').
            input_shape (Tuple[int, int, int], optional): (C, H, W). Defaults to None.
            num_classes (int, optional): Number of output classes. Defaults to None.
            pretrained (bool, optional): Load ImageNet weights. Defaults to False.

        Returns:
            nn.Module: Initialized PyTorch model on the appropriate device.
        """
        input_shape = input_shape or self.config.get('input_shape', (3, 32, 32))
        num_classes = num_classes or self.config.get('num_classes', 10)
        channels, height, width = input_shape
        weights = 'IMAGENET1K_V1' if pretrained else None

        # 1. Try to load from torchvision.models
        try:
            if hasattr(models, model_type):
                builder = getattr(models, model_type)
                try:
                    m = builder(weights=weights)
                except TypeError:
                    # Some older models might not support 'weights' argument
                    m = builder(pretrained=pretrained)
            elif model_type == 'resnet':
                m = models.resnet18(weights=weights)
            elif model_type == 'densenet':
                m = models.densenet121(weights=weights)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        except Exception as e:
            raise ValueError(f"Failed to load model '{model_type}': {e}")

        # 2. Apply Architecture-Specific Adaptations (e.g., for CIFAR)
        return self.adapt_model_to_task(m, input_shape, num_classes)

    def adapt_model_to_task(self, model: nn.Module, input_shape: Tuple[int, int, int], 
                           num_classes: int) -> nn.Module:
        """Adapts a standard model (e.g. ImageNet-sized) to specific input/output.
        
        Specifically handles CIFAR-sized (32x32) inputs by modifying early layers
        to prevent excessive downsampling.
        """
        channels, height, width = input_shape
        name = model.__class__.__name__.lower()

        # Handle ResNet Family
        if "resnet" in name:
            if height < 64:
                model.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()
            if hasattr(model, "fc"):
                model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Handle VGG Family
        elif "vgg" in name:
            if channels != 3:
                model.features[0] = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
            
            in_f = model.classifier[0].in_features
            if height < 64:
                model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                in_f = 512 # For VGG16, last features layer has 512 channels
            
            # Rebuild classifier for the specified number of classes
            if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
                model.classifier = nn.Sequential(
                    nn.Linear(in_f, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, num_classes),
                )

        # Handle DenseNet Family
        elif "densenet" in name:
            model.features.conv0 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if height < 64:
                model.features.pool0 = nn.Identity()
            if hasattr(model, "classifier"):
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)

        # Generic fallback for output layer
        else:
            for attr in ['fc', 'classifier', 'head']:
                if hasattr(model, attr):
                    layer = getattr(model, attr)
                    if isinstance(layer, nn.Linear):
                        setattr(model, attr, nn.Linear(layer.in_features, num_classes))
                    elif isinstance(layer, nn.Sequential) and isinstance(layer[-1], nn.Linear):
                        # Try to replace the last linear layer in a sequence
                        in_f = layer[-1].in_features
                        new_list = list(layer.children())
                        new_list[-1] = nn.Linear(in_f, num_classes)
                        setattr(model, attr, nn.Sequential(*new_list))

        return model.to(self.device)

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
        name = str(name or "train")
        auto_baseline = bool((self.config or {}).get("baseline_checkpoint_policy", "auto") != "off")
        if auto_baseline and self._is_baseline_name(name):
            latest = self._latest_baseline_ckpt(model)
            if latest is not None and latest.exists() and bool((self.config or {}).get("baseline_prefer_existing", True)):
                self.load_checkpoint(model, str(latest))
                print(f"[baseline] Loaded existing checkpoint: {latest}")
                hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
                return hist
        lr = float(self.config.get('lr', 3e-4))
        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = None
        if val_loader and bool(self.config.get("torch_reduce_lr_on_plateau", True)):
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="max",
                factor=float(self.config.get("torch_lr_reduce_factor", 0.5)),
                patience=max(1, int(self.config.get("torch_lr_patience", 1))),
                min_lr=float(self.config.get("torch_min_lr", 1e-6)),
            )
        crit = nn.CrossEntropyLoss()
        hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = -float("inf")
        best_state = None
        
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
                print(f"[val] Loss {v_loss:.4f}, Acc {v_acc:.2f}%")
                if scheduler is not None:
                    scheduler.step(v_acc)
                if bool(self.config.get("torch_restore_best", True)) and v_acc > best_val_acc:
                    best_val_acc = float(v_acc)
                    best_state = copy.deepcopy(model.state_dict())

        if best_state is not None and bool(self.config.get("torch_restore_best", True)):
            model.load_state_dict(best_state)
            print(f"[train] Restored best validation checkpoint (acc={best_val_acc:.2f}%).")

        if auto_baseline and self._is_baseline_name(name):
            out = self._new_baseline_ckpt(model)
            self.save_checkpoint(model, str(out))
            print(f"[baseline] Saved checkpoint: {out}")
        
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
        model.eval()
        if loader:
            try:
                x, _ = next(iter(loader))
                in_shape = tuple(x.shape[1:])
            except Exception:
                in_shape = self.config.get('input_shape', (3, 32, 32))
        else:
            in_shape = self.config.get('input_shape', (3, 32, 32))
            
        # Calculate parameters directly for 100% accuracy
        total_params = sum(p.numel() for p in model.parameters())
        
        try:
            from thop import profile
            # Using 'thop' for FLOP estimation
            device = next(model.parameters()).device
            dummy_in = torch.randn(1, *in_shape).to(device)
            f, _ = profile(model, inputs=(dummy_in,), verbose=False)
            return float(f), float(total_params)
        except Exception: 
            return 0.0, float(total_params)

    def save_checkpoint(self, model: nn.Module, path: str):
        """Saves model state dict."""
        state = model.state_dict()
        # THOP profiling can inject bookkeeping buffers (total_ops/total_params).
        # Strip them so checkpoints stay portable and reload cleanly.
        filtered = {
            k: v for k, v in state.items()
            if not (k.endswith("total_ops") or k.endswith("total_params"))
        }
        torch.save(filtered, path)
        
    def load_checkpoint(self, model: nn.Module, path: str):
        """Loads model state dict."""
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict):
            state = {
                k: v for k, v in state.items()
                if not (k.endswith("total_ops") or k.endswith("total_params"))
            }
        model.load_state_dict(state, strict=False)

    def _activation_scores(self, model: nn.Module, loader: Any, method: str) -> Dict[str, np.ndarray]:
        """Calculates bundled activation-based channel scores.

        Bundled activation methods are intentionally minimal:
        - mean_abs_act
        - apoz

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
        
        # Temporarily disable inplace operations to avoid RuntimeError with backward hooks
        inplace_mods = []
        for m in model.modules():
            if hasattr(m, "inplace") and m.inplace:
                m.inplace = False
                inplace_mods.append(m)

        try:
            # Get prunable layers and identify which ones to score
            deps = self.trace_graph(model)
            prunables = self._iter_prunable_layers(model, deps=deps)
            max_batches = self._resolve_prune_batches(loader)
            
            def get_hook(name):
                def hook(module, grad_input, grad_output):
                    o = grad_output[0] if isinstance(grad_output, tuple) else grad_output
                    # Flatten activations for Linear layers if needed (B, C, H, W) -> (B, C)
                    if o.dim() == 4:
                        reduce_dims = (0, 2, 3)
                    else:
                        reduce_dims = (0,)

                    if method == "apoz":
                        post_relu = torch.relu(o.detach())
                        apoz = (post_relu == 0).float().mean(dim=reduce_dims)
                        val = (1.0 - apoz).cpu().numpy()
                    else: 
                        val = torch.abs(o.detach()).mean(dim=reduce_dims).cpu().numpy()
                    
                    if name not in scores: 
                        scores[name] = []
                    scores[name].append(val)
                return hook
                
            for name, layer in prunables:
                hooks.append(layer.register_forward_hook(lambda m, i, o, n=name: get_hook(n)(m, None, o)))
            
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device)
                with torch.no_grad():
                    model(x)
        finally:
            for h in hooks: 
                h.remove()
            # Restore inplace operations
            for m in inplace_mods:
                m.inplace = True

        return {n: np.mean(v, axis=0) for n, v in scores.items()}

    def _single_pass_multi_metric_scores(
        self, model: nn.Module, loader: Any, metrics: List[str]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Collects bundled activation metrics in one calibration pass."""
        requested = {m.lower().strip() for m in metrics}
        supported = {"mean_abs_act", "apoz"}
        active = requested.intersection(supported)
        if not active:
            return {}

        deps = self.trace_graph(model)
        prunables = self._iter_prunable_layers(model, deps=deps)

        if not prunables:
            return {m: {} for m in active}

        model.eval()
        max_batches = self._resolve_prune_batches(loader)
        # Temporarily disable inplace operations
        inplace_mods = []
        for m in model.modules():
            if hasattr(m, "inplace") and m.inplace:
                m.inplace = False
                inplace_mods.append(m)

        # Initialize aggregators
        results_agg = {m: {n: [] for n, _ in prunables} for m in active}
        hooks = []
        def forward_hook(name: str):
            def _hook(module, _input, output):
                o = output[0] if isinstance(output, tuple) else output
                od = o.detach()
                reduce_dims = (0, 2, 3) if od.dim() == 4 else (0,)

                if "mean_abs_act" in active:
                    results_agg["mean_abs_act"][name].append(torch.abs(od).mean(dim=reduce_dims).cpu().numpy())

                if "apoz" in active:
                    post_relu = torch.relu(od)
                    apoz = (post_relu == 0).float().mean(dim=reduce_dims)
                    results_agg["apoz"][name].append((1.0 - apoz).cpu().numpy())

            return _hook

        for name, layer in prunables:
            hooks.append(layer.register_forward_hook(forward_hook(name)))

        try:
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device)
                with torch.no_grad():
                    model(x)
        finally:
            for h in hooks: h.remove()
            # Restore inplace operations
            for m in inplace_mods:
                m.inplace = True

        return {m: {n: np.mean(v, axis=0) for n, v in res.items()} for m, res in results_agg.items()}

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
        if method in ("mean_abs_act", "apoz"):
            return self._activation_scores(model, loader, method)
            
        deps = self.trace_graph(model)
        prunables = self._iter_prunable_layers(model, deps=deps)
        tools = CustomMethodTools(
            framework="torch",
            model=model,
            loader=loader,
            device=self.device,
            config=self.config,
            prunables=prunables,
        )

        score_map = {}
        for name, layer in prunables:
            fn_kwargs = {
                "layer_name": name, "layer": layer, "model": model, 
                "loader": loader, "device": self.device,
                "tools": tools, "prunables": prunables
            }
            for k, v in (self.config or {}).items():
                fn_kwargs.setdefault(k, v)
            s = call_score_fn(method, "torch", fn_kwargs)
            if s is None: continue
            if isinstance(s, torch.Tensor): s = s.detach().cpu().numpy()
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

    def trace_graph(self, model: nn.Module) -> Dict[str, Any]:
        """Traces the model structure using the internal pruner."""
        return TorchStructuralPruner(model).dependencies

    def classify_architecture(self, model: nn.Module) -> str:
        """Categorizes the model based on graph features."""
        try:
            traced = torch.fx.symbolic_trace(model)
            ops = [str(n.target) for n in traced.graph.nodes if n.op == 'call_function']
            if any('cat' in op for op in ops): return "concatenative"
            if any(x in str(op) for op in ops for x in ('add', 'iadd')): return "residual"
        except Exception: pass
        return "sequential"

    def get_multi_metric_scores(self, model: nn.Module, loader: Any, metrics: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
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
