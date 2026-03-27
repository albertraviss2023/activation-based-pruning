import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from reducnn.engine.orchestrator import Orchestrator
from reducnn.backends.factory import get_adapter

def main():
    print("=== ReduCNN: Pretrained Baseline Pipeline ===")
    
    # 1. Setup Configuration
    # We specify the pretrained checkpoint to load, and where to save the pruned model.
    # The Orchestrator will automatically handle loading and saving.
    MODEL_TYPE = 'resnet18' # options: 'resnet18', 'vgg16', etc.
    
    config = {
        'model_type': MODEL_TYPE,
        'backend': 'pytorch',
        'method': 'mean_abs_act', 
        'scope': 'global',        
        'ratio': 0.5,             
        'epochs': 0,              
        'ft_epochs': 1,           
        'pretrained_checkpoint_path': f'./checkpoints/{MODEL_TYPE}_baseline.pth',
        'pruned_checkpoint_path': f'./checkpoints/{MODEL_TYPE}_pruned_untrained.pth',
        'final_checkpoint_path': f'./checkpoints/{MODEL_TYPE}_pruned_finetuned.pth'
    }
    
    # 2. Setup Data (CIFAR-10)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Use a small subset for demonstration speed
    train_sub = torch.utils.data.Subset(trainset, range(500))
    test_sub = torch.utils.data.Subset(testset, range(100))
    
    trainloader = DataLoader(train_sub, batch_size=32, shuffle=True)
    testloader = DataLoader(test_sub, batch_size=32, shuffle=False)
    
    # 3. Create a Dummy "Pretrained" Model Checkpoint for the test
    os.makedirs('./checkpoints', exist_ok=True)
    if not os.path.exists(config['pretrained_checkpoint_path']):
        print(f"Creating a dummy pretrained {MODEL_TYPE} checkpoint to simulate user input...")
        from reducnn.backends.torch_backend import PyTorchAdapter
        adapter = PyTorchAdapter(config)
        dummy_model = adapter.get_model(MODEL_TYPE, input_shape=(3, 32, 32), num_classes=10)
        adapter.save_checkpoint(dummy_model, config['pretrained_checkpoint_path'])
    
    # 4. Initialize and Run the Orchestrator
    orchestrator = Orchestrator(config)
    
    print("\nStarting the Orchestrator pipeline...")
    # The orchestrator will automatically load the model, evaluate, prune, and fine-tune!
    pruned_model, masks = orchestrator.run(
        loader=trainloader, 
        val_loader=testloader
    )
    
    # 5. Visualize Model Architecture (Bonus)
    print("\nVisualizing Model Architecture...")
    from reducnn.visualization.animator import PruningAnimator
    adapter = get_adapter(pruned_model, config)
    animator = PruningAnimator(adapter)
    
    # Generate the static architecture plot
    fig = animator.plot_architecture(pruned_model, title=f"Pruned {MODEL_TYPE} Architecture", render=False)
    
    # Export it to an HTML file locally
    html_path = animator.export_html(fig, path=f"pruned_{MODEL_TYPE}_architecture.html")
    print(f"✅ Architecture visualization saved to: {html_path}")
    print("\nPipeline finished successfully! Check the './checkpoints' folder for saved models.")

if __name__ == '__main__':
    main()
