#!/usr/bin/env python3
"""
Integration test for save/load cycle without full dependencies.

This test simulates the real save/load workflow that would happen
between training and inference, ensuring:
1. Model saves correctly
2. Model loads without projection
3. No meta tensor errors
"""

import os
import sys
import tempfile
import torch
import json
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create comprehensive mocks for WAN modules
wan_module = MagicMock()
wan_text2video = MagicMock()
wan_configs = MagicMock()
wan_configs_t2v = MagicMock()

# Set up the module hierarchy
sys.modules['wan'] = wan_module
sys.modules['wan.text2video'] = wan_text2video
sys.modules['wan.configs'] = wan_configs
sys.modules['wan.configs.wan_t2v_A14B'] = wan_configs_t2v

# Mock specific items that will be imported
wan_text2video.WanT2V = MagicMock
wan_configs_t2v.t2v_A14B = {}

# Mock projection_mapper
projection_mapper = MagicMock()
projection_called = [False]  # Track if projection is called

def mock_load_and_project_weights(*args, **kwargs):
    print("[Mock] ⚠️  Projection called - this should NOT happen during loading!")
    projection_called[0] = True
    
projection_mapper.load_and_project_weights = mock_load_and_project_weights
sys.modules['projection_mapper'] = projection_mapper

# Now import train_distillation
from train_distillation import WanLiteStudent


def test_save_and_load_integration():
    """Test the complete save/load cycle."""
    print("=" * 80)
    print("INTEGRATION TEST: Save and Load Student Model")
    print("=" * 80)
    
    # 1. Create a student model (simulating training completion)
    print("\n[1/5] Creating student model (simulating trained model)...")
    config = {
        'model_type': 'WanLiteStudent',
        'hidden_size': 256,
        'depth': 4,
        'num_heads': 8,
        'num_channels': 4,
        'image_size': 512,
        'patch_size': 16,
        'text_max_length': 77,
        'text_encoder_output_dim': 4096,
        'projection_factor': 1.0
    }
    
    # Create model without projection (simulating post-training state)
    model = WanLiteStudent(
        config,
        teacher_checkpoint_path=None,  # No projection needed - model is already trained
        distributed=False
    )
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 2. Save the model (simulating end of training)
    print("\n[2/5] Saving model (simulating save at end of training)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "trained_model")
        model.save_pretrained(save_path)
        
        # Verify saved files
        config_path = os.path.join(save_path, "config.json")
        assert os.path.exists(config_path), "config.json not found"
        
        # Check for weights file (either naming convention)
        weights_files = [
            "diffusion_pytorch_model.safetensors",
            "diffusion_model.safetensors"
        ]
        weights_found = any(os.path.exists(os.path.join(save_path, f)) for f in weights_files)
        assert weights_found, "No weights file found"
        
        print("✓ Model saved successfully")
        print(f"   Location: {save_path}")
        print(f"   Files: {os.listdir(save_path)}")
        
        # 3. Simulate starting a new Python session (inference mode)
        print("\n[3/5] Simulating inference mode (new Python session)...")
        print("   This is where the error would occur with old code...")
        
        # 4. Load the model for inference
        print("\n[4/5] Loading model for inference...")
        print("   IMPORTANT: No projection should happen here!")
        print("   Watch for '[Mock] ⚠️ Projection called' messages (there should be NONE)")
        print()
        
        # Reset projection tracker
        projection_called[0] = False
        
        try:
            loaded_model = WanLiteStudent.from_pretrained(save_path)
            print()
            
            if projection_called[0]:
                print("✗ ERROR: Projection was triggered during loading!")
                print("   This means the fix didn't work properly.")
                return False
            
            print("✓ Model loaded successfully for inference")
            print("   No projection was triggered (correct!)")
        except Exception as e:
            print()
            print(f"✗ ERROR during loading: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 5. Verify the loaded model
        print("\n[5/5] Verifying loaded model...")
        
        # Check that it's in eval mode by default or can be set
        loaded_model.eval()
        
        # Check parameter count matches
        orig_params = sum(p.numel() for p in model.parameters())
        loaded_params = sum(p.numel() for p in loaded_model.parameters())
        assert orig_params == loaded_params, f"Parameter count mismatch: {orig_params} vs {loaded_params}"
        print(f"✓ Parameter count matches: {loaded_params}")
        
        # Check config matches
        for key in ['hidden_size', 'depth', 'num_heads', 'num_channels']:
            orig_val = config[key]
            loaded_val = getattr(loaded_model.config, key)
            assert orig_val == loaded_val, f"Config mismatch for {key}: {orig_val} vs {loaded_val}"
        print("✓ Configuration matches")
        
        # Check weights match (check multiple parameters for robustness)
        print("   Checking weight integrity across multiple layers...")
        orig_params = list(model.parameters())
        loaded_params = list(loaded_model.parameters())
        
        # Use dtype-appropriate tolerance
        rtol = 1e-4 if orig_params[0].dtype == torch.float32 else 1e-3
        
        # Check first, middle, and last parameters
        params_to_check = [0, len(orig_params) // 2, -1]
        for idx in params_to_check:
            if not torch.allclose(orig_params[idx], loaded_params[idx], rtol=rtol):
                print(f"✗ Weights don't match at parameter index {idx}!")
                return False
        print(f"✓ Weights match (checked first, middle, and last parameters, rtol={rtol})")
        
    print("\n" + "=" * 80)
    print("✅ INTEGRATION TEST PASSED!")
    print("=" * 80)
    print("\nThe fix successfully resolves:")
    print("  1. ✓ No projection happens during model loading for inference")
    print("  2. ✓ No meta tensor errors during loading")
    print("  3. ✓ Model loads correctly with proper weights and config")
    print("  4. ✓ Ready for inference without teacher model")
    print()
    return True


if __name__ == "__main__":
    print("\n🧪 Running Save/Load Integration Test\n")
    success = test_save_and_load_integration()
    sys.exit(0 if success else 1)
