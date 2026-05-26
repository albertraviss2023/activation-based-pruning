import os
import torch
import shutil
from reducnn.backends.factory import get_adapter

def test_model_persistence():
    print("🚀 Testing Model Persistence (Save/Load) logic...")
    
    SAVE_DIR = "test_saved_models"
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    config = {
        'backend': 'pytorch',
        'model_type': 'vgg11',
        'input_shape': (3, 32, 32),
        'num_classes': 10
    }
    
    # 1. Initialize Adapter
    from reducnn.backends.torch_backend import PyTorchAdapter
    adapter = PyTorchAdapter(config)
    model = adapter.get_model('vgg11', pretrained=False)
    
    # 2. Define Path
    model_path = os.path.join(SAVE_DIR, "test_vgg11.pth")
    
    # 3. Test Saving
    print(f"  Saving model to {model_path}...")
    adapter.save_checkpoint(model, model_path)
    if os.path.exists(model_path):
        print("  ✅ File created successfully.")
    else:
        raise FileNotFoundError("❌ Failed to create model file.")
        
    # 4. Test Loading
    print("  Loading model back...")
    new_model = adapter.get_model('vgg11', pretrained=False)
    # Check that it doesn't crash
    adapter.load_checkpoint(new_model, model_path)
    print("  ✅ Model loaded successfully.")
    
    # Clean up
    shutil.rmtree(SAVE_DIR)
    print("\n🚀 PERSISTENCE TESTS PASSED!")

if __name__ == "__main__":
    test_model_persistence()
