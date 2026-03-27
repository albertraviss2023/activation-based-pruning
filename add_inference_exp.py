import json
import os

def add_inference_diagnostics(file_path, class_names=None):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # 1. Add Markdown Header
    h_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Part D: Performance & Inference Diagnostics\n",
            "This section compares the **real-world inference speed** and **prediction quality** of the Original vs. Pruned models."
        ]
    }
    
    # 2. Add Benchmarking Code
    b_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import time\n",
            "import torch\n",
            "\n",
            "def benchmark_inference(model, loader, device, iterations=100):\n",
            "    model.eval()\n",
            "    # Extract a single sample for latency test\n",
            "    it = iter(loader)\n",
            "    x, _ = next(it)\n",
            "    x = x[:1].to(device)\n",
            "    \n",
            "    # Warm-up\n",
            "    with torch.no_grad():\n",
            "        for _ in range(20): _ = model(x)\n",
            "    \n",
            "    # Timing\n",
            "    start = time.time()\n",
            "    with torch.no_grad():\n",
            "        for _ in range(iterations):\n",
            "            _ = model(x)\n",
            "    end = time.time()\n",
            "    \n",
            "    latency = (end - start) / iterations * 1000 # ms\n",
            "    return latency\n",
            "\n",
            "print('⏱️ Benchmarking Latency (Batch Size = 1)...')\n",
            "t_orig = benchmark_inference(orig_model, loader, adapter.device)\n",
            "t_pruned = benchmark_inference(pruned_model, loader, adapter.device)\n",
            "\n",
            "print(f'   Original {MODEL_TYPE} Latency: {t_orig:.3f} ms')\n",
            "print(f'   Pruned {MODEL_TYPE} Latency:   {t_pruned:.3f} ms')\n",
            "print(f'   🚀 Speedup: {(t_orig/t_pruned):.2f}x')"
        ]
    }
    
    # 3. Add Visualization Code
    v_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('🖼️ Generating Inference Gallery...')\n",
            "it = iter(loader)\n",
            "images, labels = next(it)\n",
            "images_sub = images[:8]\n",
            "labels_sub = labels[:8]\n",
            "\n",
            "with torch.no_grad():\n",
            "    p_orig = torch.argmax(orig_model(images_sub.to(adapter.device)), dim=1).cpu().numpy()\n",
            "    p_pruned = torch.argmax(pruned_model(images_sub.to(adapter.device)), dim=1).cpu().numpy()\n",
            "\n",
            f"class_names = {class_names}\n",
            "viz.plot_inference_gallery(images_sub.numpy(), labels_sub.numpy(), \n",
            "                           p_orig, p_pruned, \n",
            "                           class_names=class_names, title=f'{MODEL_TYPE}: Original vs. Pruned Predictions')"
        ]
    }
    
    nb['cells'].extend([h_cell, b_cell, v_cell])
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully added diagnostics to {file_path}")

if __name__ == "__main__":
    c10_labels = "['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']"
    c_dog_labels = "['cat', 'dog']"
    
    add_inference_diagnostics('experiments.ipynb', class_names=c10_labels)
    add_inference_diagnostics('experiments_cat_dog.ipynb', class_names=c_dog_labels)
    add_inference_diagnostics('experiments_cifar100.ipynb', class_names=None)
