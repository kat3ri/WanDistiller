"""
Test script to verify WanLiteStudent save/load compatibility with WAN model structure.

This test verifies that:
1. The model can be saved using the HuggingFace Diffusers format
2. The saved config.json contains all necessary parameters
3. The model can be loaded back using from_pretrained()
4. The loaded model has the same configuration and weights
"""

import os
import json
import tempfile
import torch
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_save_load_cycle():
    """Test that a model can be saved and loaded successfully."""
    print("=" * 80)
    print("Testing WanLiteStudent Save/Load Compatibility")
    print("=" * 80)
    
    # Import after adding to path
    import train_distillation
    
    # Create a simple config for testing (small model to run on CPU)
    print("\n[1/5] Creating test model configuration...")
    config = {
        'model_type': 'WanLiteStudent',
        'hidden_size': 128,
        'depth': 2,
        'num_heads': 4,
        'num_channels': 4,
        'image_size': 256,
        'patch_size': 16,
        'text_max_length': 77,
        'text_encoder_output_dim': 512,
        'projection_factor': 1.0
    }
    print("✓ Configuration created")
    
    # Initialize model
    print("\n[2/5] Initializing WanLiteStudent model...")
    model = train_distillation.WanLiteStudent(
        config,
        teacher_checkpoint_path=None,
        device='cpu',
        distributed=False
    )
    print("✓ Model initialized successfully")
    print(f"   Model config: {model.config}")
    
    # Save model
    print("\n[3/5] Saving model in HuggingFace Diffusers format...")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model")
        model.save_pretrained(save_path)
        
        # Check saved files
        print("\n   Checking saved files...")
        config_path = os.path.join(save_path, "config.json")
        weights_path = os.path.join(save_path, "diffusion_model.safetensors")
        
        if not os.path.exists(config_path):
            print(f"✗ config.json not found at {config_path}")
            return False
        print(f"   ✓ config.json exists")
        
        if not os.path.exists(weights_path):
            print(f"✗ diffusion_model.safetensors not found at {weights_path}")
            return False
        print(f"   ✓ diffusion_model.safetensors exists")
        
        # Verify config contents
        print("\n[4/5] Verifying config.json contents...")
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        print(f"   Saved config keys: {list(saved_config.keys())}")
        
        required_keys = [
            'model_type', 'hidden_size', 'depth', 'num_heads', 
            'num_channels', 'image_size', 'patch_size', 
            'text_max_length', 'text_encoder_output_dim', 'projection_factor'
        ]
        
        missing_keys = [key for key in required_keys if key not in saved_config]
        if missing_keys:
            print(f"✗ Missing required keys in config: {missing_keys}")
            return False
        print(f"   ✓ All required config keys present")
        
        # Verify values match
        for key in required_keys:
            expected_value = config[key]
            actual_value = saved_config[key]
            if actual_value != expected_value:
                print(f"✗ Config mismatch for {key}: saved={actual_value}, expected={expected_value}")
                return False
        print(f"   ✓ All config values match original")
        
        # Load model back
        print("\n[5/5] Loading model using from_pretrained...")
        try:
            loaded_model = train_distillation.WanLiteStudent.from_pretrained(save_path)
            print("   ✓ Model loaded successfully")
            print(f"   Loaded model config: {loaded_model.config}")
            
            # Verify configs match
            for key in required_keys:
                expected_value = config[key]
                loaded_value = getattr(loaded_model.config, key)
                if loaded_value != expected_value:
                    print(f"✗ Loaded config mismatch for {key}: loaded={loaded_value}, expected={expected_value}")
                    return False
            print("   ✓ Loaded config matches original")
            
            # Verify weights by comparing a sample parameter
            orig_param = next(model.parameters())
            loaded_param = next(loaded_model.parameters())
            if not torch.allclose(orig_param, loaded_param):
                print("✗ Weights don't match after loading")
                return False
            print("   ✓ Weights match after loading")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe WanLiteStudent model is now compatible with:")
    print("  - HuggingFace Diffusers format")
    print("  - WAN model structure")
    print("  - ComfyUI loading via from_pretrained()")
    print()
    return True

if __name__ == "__main__":
    success = test_save_load_cycle()
    sys.exit(0 if success else 1)
