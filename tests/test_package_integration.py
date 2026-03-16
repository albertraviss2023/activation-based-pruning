from __future__ import annotations
import os
import sys
import numpy as np
import pytest
import torch
import tensorflow as tf

# Add src to path for direct package testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from surgical_pruning import CloudStorage
from surgical_pruning.pruner import SurgicalPruner
from surgical_pruning.backends.torch_backend import PyTorchAdapter
from surgical_pruning.backends.keras_backend import KerasAdapter
from surgical_pruning.engine import Orchestrator

def test_cloud_storage_resolution():
    storage = CloudStorage(project_name="test_proj")
    # Should resolve to local path when not in Colab
    path = storage.resolve_path("my_models", use_drive=True)
    assert "test_proj" not in str(path) if not storage.is_colab else "test_proj" in str(path)
    assert os.path.exists(path)

def test_pytorch_adapter_integration(tmp_path):
    cfg = {"lr": 1e-3, "model_type": "vgg16"}
    adapter = PyTorchAdapter(cfg)
    model = adapter.get_model("vgg16")
    
    # Verify surgery components
    assert isinstance(model, torch.nn.Module)
    stats = adapter.get_stats(model)
    assert stats[1] > 0 # Parameters

def test_keras_adapter_integration():
    cfg = {"lr": 1e-3}
    adapter = KerasAdapter(cfg)
    model = adapter.get_model("vgg16")
    
    assert isinstance(model, tf.keras.Model)
    stats = adapter.get_stats(model)
    assert stats[1] > 0

def test_surgical_pruner_dispatch():
    # Test that the pruner automatically detects PyTorch
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU()
    )
    # Mock loader
    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))
    loader = [(x, y)]
    
    surgeon = SurgicalPruner(method='l1_norm')
    # This triggers @framework_dispatch
    pruned, masks = surgeon.prune(model, loader, ratio=0.5)
    
    assert len(masks) > 0
    assert "0" in masks # Sequential names layers by index
