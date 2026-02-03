#!/usr/bin/env python3
"""
Test script to verify HuggingFace format detection logic (standalone, no GPU needed).
"""

import os
import sys
import tempfile


def _is_huggingface_format(checkpoint_dir):
    """
    Detect if the checkpoint directory is in HuggingFace Diffusers format.
    (Copied from wan/text2video.py to avoid GPU initialization issues)
    """
    # If it contains a "/" and doesn't exist locally, it might be a HF model ID
    if '/' in checkpoint_dir and not os.path.exists(checkpoint_dir):
        return True
    
    # Check if directory exists
    if not os.path.exists(checkpoint_dir):
        # If it doesn't exist, assume it's a HF model ID
        return True
    
    # Check for HuggingFace structure markers
    hf_markers = ['text_encoder', 'vae', 'transformer', 'transformer_2', 'tokenizer']
    has_hf_markers = sum(1 for marker in hf_markers if os.path.exists(os.path.join(checkpoint_dir, marker))) >= 3
    
    # Check for local format markers
    local_markers = ['models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth', 'low_noise_model', 'high_noise_model']
    has_local_markers = sum(1 for marker in local_markers if os.path.exists(os.path.join(checkpoint_dir, marker))) >= 2
    
    # If we have HF markers but not local markers, it's HF format
    if has_hf_markers and not has_local_markers:
        return True
    
    # Default to local format for backward compatibility
    return False


def test_format_detection():
    """Test the format detection logic with mock directory structures."""
    
    print("Testing format detection logic...")
    print("=" * 80)
    
    # Test 1: HuggingFace model ID (with /)
    print("\n1. Testing HuggingFace model ID format:")
    test_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    is_hf = _is_huggingface_format(test_path)
    print(f"   Path: {test_path}")
    print(f"   Result: {'HuggingFace' if is_hf else 'Local'} format")
    print(f"   ✓ PASS" if is_hf else "   ✗ FAIL")
    assert is_hf, "Should detect HF format for model ID with /"
    
    # Test 2: Non-existent path (assume HF model ID)
    print("\n2. Testing non-existent path:")
    test_path = "some-org/some-model"
    is_hf = _is_huggingface_format(test_path)
    print(f"   Path: {test_path}")
    print(f"   Result: {'HuggingFace' if is_hf else 'Local'} format")
    print(f"   ✓ PASS" if is_hf else "   ✗ FAIL")
    assert is_hf, "Should detect HF format for non-existent path with /"
    
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 3: Local format with .pth files
        print("\n3. Testing local checkpoint format:")
        local_dir = os.path.join(tmpdir, "local_checkpoint")
        os.makedirs(local_dir)
        
        # Create marker files for local format
        open(os.path.join(local_dir, "models_t5_umt5-xxl-enc-bf16.pth"), 'w').close()
        open(os.path.join(local_dir, "Wan2.1_VAE.pth"), 'w').close()
        os.makedirs(os.path.join(local_dir, "low_noise_model"))
        os.makedirs(os.path.join(local_dir, "high_noise_model"))
        
        is_hf = _is_huggingface_format(local_dir)
        print(f"   Path: {local_dir}")
        print(f"   Result: {'HuggingFace' if is_hf else 'Local'} format")
        print(f"   ✓ PASS" if not is_hf else "   ✗ FAIL")
        assert not is_hf, "Should detect local format for directory with .pth files"
        
        # Test 4: HuggingFace format with subfolders
        print("\n4. Testing HuggingFace diffusers format:")
        hf_dir = os.path.join(tmpdir, "hf_checkpoint")
        os.makedirs(hf_dir)
        
        # Create marker directories for HF format
        os.makedirs(os.path.join(hf_dir, "text_encoder"))
        os.makedirs(os.path.join(hf_dir, "vae"))
        os.makedirs(os.path.join(hf_dir, "transformer"))
        os.makedirs(os.path.join(hf_dir, "transformer_2"))
        os.makedirs(os.path.join(hf_dir, "tokenizer"))
        
        is_hf = _is_huggingface_format(hf_dir)
        print(f"   Path: {hf_dir}")
        print(f"   Result: {'HuggingFace' if is_hf else 'Local'} format")
        print(f"   ✓ PASS" if is_hf else "   ✗ FAIL")
        assert is_hf, "Should detect HF format for directory with HF subfolders"
        
        # Test 5: Mixed format (both HF and local markers) - should prefer local
        print("\n5. Testing mixed format (has both):")
        mixed_dir = os.path.join(tmpdir, "mixed_checkpoint")
        os.makedirs(mixed_dir)
        
        # Create both local and HF markers
        open(os.path.join(mixed_dir, "models_t5_umt5-xxl-enc-bf16.pth"), 'w').close()
        open(os.path.join(mixed_dir, "Wan2.1_VAE.pth"), 'w').close()
        os.makedirs(os.path.join(mixed_dir, "text_encoder"))
        os.makedirs(os.path.join(mixed_dir, "vae"))
        
        is_hf = _is_huggingface_format(mixed_dir)
        print(f"   Path: {mixed_dir}")
        print(f"   Result: {'HuggingFace' if is_hf else 'Local'} format")
        print(f"   ✓ PASS (defaults to local for backward compatibility)" if not is_hf else "   ✗ FAIL")
        assert not is_hf, "Should default to local format when both markers present"
    
    print("\n" + "=" * 80)
    print("✓ All format detection tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_format_detection()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
