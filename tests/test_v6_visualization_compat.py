from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from torch.utils.data import DataLoader, TensorDataset

from tests.notebook_utils import load_notebook_namespace


def _load_v6():
    return load_notebook_namespace(
        "generalized_pruning_v6_final.ipynb",
        "# =========================================\n# 5. EXECUTION",
    )


def test_v6_visualization_functions_still_run():
    v6 = _load_v6()
    torch = pytest.importorskip("torch")

    class TinyConvNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(8, 8, 3, padding=1)
            self.conv3 = torch.nn.Conv2d(8, 8, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(8, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = TinyConvNet()
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    # V6 visualize_activations uses a global `device` variable.
    v6.device = torch.device("cpu")

    pruned_masks = {
        "conv1": torch.rand(8) > 0.3,
        "conv2": torch.rand(8) > 0.3,
        "conv3": torch.rand(8) > 0.3,
    }

    v6.visualize_filters(model, "V6 Filters Compatibility")
    v6.visualize_activations(model, loader, "V6 Activations Compatibility", num_layers=2)
    v6.visualize_all_layers_heatmaps(None, pruned_masks)

