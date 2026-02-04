"""
Test script to verify ComfyUI compatibility for WanLiteStudent models.

This test verifies that the saved config.json includes all metadata
fields required by ComfyUI to properly load and instantiate the model.
"""

import os
import json
import tempfile
import sys
import unittest.mock as mock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_comfy_metadata():
    """Test that saved config.json includes ComfyUI-required metadata."""
    print("=" * 80)
    print("Testing ComfyUI Compatibility Metadata")
    print("=" * 80)
    
    # Mock CUDA to avoid GPU requirement
    with mock.patch('torch.cuda.is_available', return_value=False):
        with mock.patch('torch.cuda.current_device', return_value=0):
            # Import after adding to path
            import train_distillation
            
            # Create a simple config for testing (small model to run on CPU)
            print("\n[1/4] Creating test model configuration...")
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
            print("\n[2/4] Initializing WanLiteStudent model...")
            model = train_distillation.WanLiteStudent(
                config,
                teacher_checkpoint_path=None,
                device='cpu',
                distributed=False
            )
            print("✓ Model initialized successfully")
            
            # Save model and check metadata
            print("\n[3/4] Saving model and checking ComfyUI metadata...")
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = os.path.join(tmpdir, "test_model")
                model.save_pretrained(save_path)
                
                # Load and verify config.json
                config_path = os.path.join(save_path, "config.json")
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                
                print("\n   Saved config.json contents:")
                print("   " + "-" * 70)
                for key, value in saved_config.items():
                    if isinstance(value, (str, int, float)):
                        print(f"   {key}: {value}")
                print("   " + "-" * 70)
                
                # Check for ComfyUI-required fields
                print("\n[4/4] Verifying ComfyUI-required metadata fields...")
                
                required_fields = {
                    '_class_name': 'WanLiteStudent',
                    '_diffusers_version': None,  # Should exist but value can vary
                }
                
                all_passed = True
                for field, expected_value in required_fields.items():
                    if field not in saved_config:
                        print(f"   ✗ MISSING: {field}")
                        all_passed = False
                    elif expected_value is not None and saved_config[field] != expected_value:
                        print(f"   ✗ INCORRECT: {field} = {saved_config[field]}, expected {expected_value}")
                        all_passed = False
                    else:
                        print(f"   ✓ {field}: {saved_config[field]}")
                
                if not all_passed:
                    print("\n" + "=" * 80)
                    print("✗ TEST FAILED - Missing required metadata for ComfyUI")
                    print("=" * 80)
                    return False
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe saved config.json includes all ComfyUI-required metadata:")
    print("  - _class_name: Tells ComfyUI which class to instantiate")
    print("  - _diffusers_version: Version tracking for compatibility")
    print("\nModels saved with this format should load correctly in ComfyUI")
    print("without the 'NoneType' object has no attribute 'clone' error.")
    print()
    return True


if __name__ == "__main__":
    success = test_comfy_metadata()
    sys.exit(0 if success else 1)
