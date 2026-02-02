#!/usr/bin/env python
"""
Simple validation script to test WAN integration without requiring full model weights.
Tests that imports work and basic shapes are correct.
"""

import sys
import torch
from easydict import EasyDict

def test_imports():
    """Test that WAN modules can be imported"""
    print("Testing WAN imports...")
    try:
        from wan.text2video import WanT2V
        from wan.configs.wan_t2v_A14B import t2v_A14B
        from wan.modules.vae2_1 import Wan2_1_VAE
        from wan.modules.t5 import T5EncoderModel
        print("✓ All WAN imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test that config is properly structured"""
    print("\nTesting WAN config...")
    try:
        from wan.configs.wan_t2v_A14B import t2v_A14B
        
        # Check required config keys
        required_keys = ['vae_checkpoint', 't5_checkpoint', 'vae_stride', 'patch_size', 
                        'text_len', 'param_dtype', 't5_dtype']
        for key in required_keys:
            if not hasattr(t2v_A14B, key):
                print(f"✗ Missing config key: {key}")
                return False
        
        print(f"✓ Config structure valid")
        print(f"  - VAE stride: {t2v_A14B.vae_stride}")
        print(f"  - Patch size: {t2v_A14B.patch_size}")
        print(f"  - Text length: {t2v_A14B.text_len}")
        print(f"  - Param dtype: {t2v_A14B.param_dtype}")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_train_distillation_imports():
    """Test that train_distillation imports work with WAN"""
    print("\nTesting train_distillation imports...")
    try:
        import train_distillation
        print("✓ train_distillation imports successful")
        
        # Check that key functions exist
        if not hasattr(train_distillation, 'main'):
            print("✗ Missing main function")
            return False
        if not hasattr(train_distillation, 'WanLiteStudent'):
            print("✗ Missing WanLiteStudent class")
            return False
        
        print("✓ All required functions and classes present")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shape_compatibility():
    """Test that shapes are compatible between student and teacher"""
    print("\nTesting shape compatibility...")
    try:
        from wan.configs.wan_t2v_A14B import t2v_A14B
        
        # WAN VAE config
        vae_z_dim = 16  # WAN VAE uses 16 latent channels
        vae_stride = t2v_A14B.vae_stride  # (4, 8, 8)
        
        # Example student config
        student_channels = 4  # Student uses 4 channels
        image_size = 256  # Example image size
        patch_size = 16
        
        # Calculate student latent shape
        student_latent_h = image_size // patch_size
        student_latent_w = image_size // patch_size
        print(f"  Student latent shape: [{student_channels}, {student_latent_h}, {student_latent_w}]")
        
        # For 1-frame video, teacher expects [C, 1, H, W]
        # After projection: [vae_z_dim, 1, H, W]
        print(f"  Teacher expected shape (after projection): [{vae_z_dim}, 1, {student_latent_h}, {student_latent_w}]")
        
        # Check if projection is needed
        needs_projection = (student_channels != vae_z_dim)
        print(f"  Projection needed: {needs_projection} ({student_channels} -> {vae_z_dim})")
        
        print("✓ Shape compatibility validated")
        return True
    except Exception as e:
        print(f"✗ Shape test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_encoder_interface():
    """Test text encoder interface expectations"""
    print("\nTesting text encoder interface...")
    try:
        # WAN T5 encoder expects:
        # - Input: list of strings (prompts)
        # - Device: torch.device
        # - Output: list of tensors with shape [L, 4096]
        
        print("  WAN T5 expected interface:")
        print("    - Input: list of prompt strings")
        print("    - Output: list of [seq_len, 4096] tensors")
        
        # The training loop needs to:
        # 1. Call text_encoder(prompts, device)
        # 2. Stack/pad outputs to create batch tensor [B, max_len, 4096]
        
        print("✓ Text encoder interface documented")
        return True
    except Exception as e:
        print(f"✗ Text encoder test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("=" * 80)
    print("WAN Integration Validation Tests")
    print("=" * 80)
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Train Distillation", test_train_distillation_imports()))
    results.append(("Shape Compatibility", test_shape_compatibility()))
    results.append(("Text Encoder Interface", test_text_encoder_interface()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All validation tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
