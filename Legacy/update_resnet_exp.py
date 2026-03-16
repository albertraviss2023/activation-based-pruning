import json
import os

def update_resnet_section(file_path, dataset_name, img_size, num_classes, is_keras_32x32=True):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}, not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    for cell in nb['cells']:
        # Detect the ResNet cells and replace them with the full workflow
        if 'id' in cell and (cell['id'] == 'torch_resnet_pruning' or cell['id'] == 'resnet_experiment'):
            # 1. PyTorch ResNet Full Workflow
            c1 = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [ f"### C.1 PyTorch ResNet-18: Full Research Workflow ({dataset_name})\n" ]
            }
            c2 = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"print('🧪 PYTORCH: ResNet-18 Full Suite ({dataset_name})')\n",
                    f"t_res_adapter = PyTorchAdapter(config={{'lr': 1e-4, 'input_shape': (3, {img_size}, {img_size}), 'num_classes': {num_classes}}})\n",
                    "t_res_model = t_res_adapter.get_model('resnet18')\n",
                    "\n",
                    "print('1. Establishing ResNet Baseline...')\n",
                    "t_res_adapter.train(t_res_model, train_loader, epochs=2, name='ResNet_Base', val_loader=test_loader)\n",
                    "res_base_acc = t_res_adapter.evaluate(t_res_model, test_loader)\n",
                    "\n",
                    "print('\\n2. Performing Structural Surgery (Local 20%)...')\n",
                    "res_surgeon = ReduCNNPruner(method='l1_norm', scope='local')\n",
                    "pruned_res, res_masks, _ = res_surgeon.prune(t_res_model, train_loader, ratio=0.2)\n",
                    "\n",
                    "print('\\n3. Healing Phase (Fine-tuning)...')\n",
                    "t_res_adapter.train(pruned_res, train_loader, epochs=3, name='ResNet_Heal', val_loader=test_loader)\n",
                    "res_pruned_acc = t_res_adapter.evaluate(pruned_res, test_loader)\n",
                    "\n",
                    "print(f'\\n✅ ResNet-18 Results ({dataset_name}):')\n",
                    "print(f'   Baseline Acc: {res_base_acc:.2f}%')\n",
                    "print(f'   Pruned Acc:   {res_pruned_acc:.2f}%')\n",
                    "viz.plot_layer_sensitivity(res_masks, f'ResNet-18 Pruning Sensitivity ({dataset_name})')"
                ]
            }
            new_cells.append(c1)
            new_cells.append(c2)
            continue

        if 'id' in cell and cell['id'] == 'keras_resnet_pruning':
            # 2. Keras ResNet Full Workflow
            k_img_shape = f"({img_size}, {img_size}, 3)"
            c1 = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [ f"### C.2 Keras ResNet-50: Full Research Workflow ({dataset_name})\n" ]
            }
            c2 = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"print('🧪 KERAS: ResNet-50 Full Suite ({dataset_name})')\n",
                    f"k_res_adapter = KerasAdapter(config={{'lr': 1e-4, 'input_shape': {k_img_shape}, 'num_classes': {num_classes}}})\n",
                    "k_res_model = k_res_adapter.get_model('resnet')\n",
                    "\n",
                    "print('1. Establishing ResNet-50 Baseline...')\n",
                    "k_res_adapter.train(k_res_model, train_loader, epochs=2, name='Keras_Res_Base', val_loader=test_loader)\n",
                    "k_res_base_acc = k_res_adapter.evaluate(k_res_model, test_loader)\n",
                    "\n",
                    "print('\\n2. Performing Structural Surgery (Local 20%)...')\n",
                    "k_res_surgeon = ReduCNNPruner(method='l1_norm', scope='local')\n",
                    "k_pruned_res, k_res_masks, _ = k_res_surgeon.prune(k_res_model, train_loader, ratio=0.2)\n",
                    "\n",
                    "print('\\n3. Healing Phase (Fine-tuning)...')\n",
                    "k_res_adapter.train(k_pruned_res, train_loader, epochs=2, name='Keras_Res_Heal', val_loader=test_loader)\n",
                    "k_res_pruned_acc = k_res_adapter.evaluate(k_pruned_res, test_loader)\n",
                    "\n",
                    "print(f'\\n✅ Keras ResNet-50 Results ({dataset_name}):')\n",
                    "print(f'   Baseline Acc: {k_res_base_acc:.2f}%')\n",
                    "print(f'   Pruned Acc:   {k_res_pruned_acc:.2f}%')\n",
                    "viz.plot_layer_sensitivity(k_res_masks, f'Keras ResNet-50 Pruning Sensitivity ({dataset_name})')"
                ]
            }
            new_cells.append(c1)
            new_cells.append(c2)
            continue

        new_cells.append(cell)

    nb['cells'] = new_cells
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully updated {file_path}")

if __name__ == "__main__":
    update_resnet_section('experiments.ipynb', 'CIFAR-10', 32, 10)
    update_resnet_section('experiments_cat_dog.ipynb', 'Cats vs Dogs', 128, 2)
    update_resnet_section('experiments_cifar100.ipynb', 'CIFAR-100', 32, 100)
