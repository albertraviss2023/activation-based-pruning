"""
Example of using the ReduCNN package in a decoupled way.
"""
import sys
import os

# Ensure the src directory is in the path for the example to run without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import reducnn.pruner as pruner
import reducnn.visualization as viz
import reducnn.analyzer as analyzer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Create a dummy model and data
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = TinyNet()
x = torch.randn(10, 3, 32, 32)
y = torch.randint(0, 10, (10,))
loader = DataLoader(TensorDataset(x, y), batch_size=5)

# 2. (Optional) Run Pareto Analysis
print("\n=== Running Pareto Analysis ===")
pareto = analyzer.ParetoAnalyzer(method='l1_norm')
# We mock it to just print since we don't want to block CI with plots
# In reality: pareto.run(model, loader, ratios=[0.2, 0.4])
print("Pareto initialized successfully.")

# 3. Perform the actual pruning (Core Engine)
print("\n=== Executing ReduCNN ===")
# Note: User doesn't need to specify "PyTorch" - the engine detects it!
surgeon = pruner.ReduCNNPruner(method='l1_norm', scope='local')
pruned_model, masks = surgeon.prune(model, loader, ratio=0.5)

print("\nOriginal Model params:", sum(p.numel() for p in model.parameters()))
print("Pruned Model params:", sum(p.numel() for p in pruned_model.parameters()))

# 4. (Optional) Visualize Stakeholder Metrics
print("\n=== Generating Stakeholder Visuals ===")
# You pass only the masks, no need for the full framework adapter
print("Masks structure generated:", list(masks.keys()))
# viz.plot_layer_sensitivity(masks, title_prefix="TinyNet")
print("Decoupled pipeline completed successfully!")
