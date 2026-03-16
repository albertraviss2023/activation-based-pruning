import os
import sys
import torch
import numpy as np
import pytest
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

from reducnn.backends.torch_backend import PyTorchAdapter
from reducnn.pruner import ReduCNNPruner

def mock_loader(batch_size, img_size, num_classes):
    """Creates a mock loader for testing without downloading datasets."""
    x = torch.randn(batch_size, 3, img_size, img_size)
    y = torch.randint(0, num_classes, (batch_size,))
    return [(x, y)]

def test_cat_dog_experiment_logic():
    """Simulates logic from experiments_cat_dog.ipynb."""
    img_size = 128
    num_classes = 2
    
    adapter = PyTorchAdapter(config={'lr': 1e-4, 'input_shape': (3, img_size, img_size), 'num_classes': num_classes})
    loader = mock_loader(2, img_size, num_classes)
    
    # 1. Test VGG16 Pruning for Cat/Dog
    model = adapter.get_model("vgg16")
    surgeon = ReduCNNPruner(method='l1_norm', scope='local')
    pruned_vgg, masks, dur = surgeon.prune(model, loader, ratio=0.3)
    
    assert adapter.get_stats(pruned_vgg)[1] < adapter.get_stats(model)[1]
    
    # 2. Test ResNet-18 Pruning for Cat/Dog
    res_model = adapter.get_model("resnet18")
    pruned_res, res_masks, res_dur = surgeon.prune(res_model, loader, ratio=0.2)
    
    # Check that it didn't crash and actually reduced parameters
    assert adapter.get_stats(pruned_res)[1] < adapter.get_stats(res_model)[1]
    # Check that output features still match num_classes
    assert pruned_res.fc.out_features == num_classes

def test_cifar100_experiment_logic():
    """Simulates logic from experiments_cifar100.ipynb."""
    img_size = 32
    num_classes = 100
    
    adapter = PyTorchAdapter(config={'lr': 1e-3, 'input_shape': (3, img_size, img_size), 'num_classes': num_classes})
    loader = mock_loader(2, img_size, num_classes)
    
    # 1. Test VGG16 Pruning for CIFAR-100
    model = adapter.get_model("vgg16")
    surgeon = ReduCNNPruner(method='l1_norm', scope='local')
    pruned_vgg, masks, dur = surgeon.prune(model, loader, ratio=0.3)
    
    assert pruned_vgg.classifier[-1].out_features == num_classes
    
    # 2. Test ResNet-18 Pruning for CIFAR-100
    res_model = adapter.get_model("resnet18")
    pruned_res, res_masks, res_dur = surgeon.prune(res_model, loader, ratio=0.2)
    
    assert pruned_res.fc.out_features == num_classes
    assert adapter.get_stats(pruned_res)[1] < adapter.get_stats(res_model)[1]

if __name__ == "__main__":
    # If run directly, run the tests
    pytest.main([__file__])
