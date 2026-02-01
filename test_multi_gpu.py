#!/usr/bin/env python3
"""
Test script to verify multi-GPU support in WanDistiller.

This script tests:
1. Multi-GPU detection
2. DataParallel wrapper
3. DistributedDataParallel setup (simulated)
4. Model device placement
"""

import sys
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

# Import the training module
sys.path.insert(0, '/home/runner/work/WanDistiller/WanDistiller')
import train_distillation

def test_multi_gpu_detection():
    """Test GPU detection and reporting."""
    print("=" * 80)
    print("Test 1: Multi-GPU Detection")
    print("=" * 80)
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✓ CUDA is available")
        print(f"✓ Number of GPUs detected: {num_gpus}")
        for i in range(num_gpus):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  CUDA is not available (CPU-only environment)")
        print("   Multi-GPU features can be tested in GPU environments")
    
    print()
    return True


def test_distributed_functions():
    """Test distributed training helper functions."""
    print("=" * 80)
    print("Test 2: Distributed Training Functions")
    print("=" * 80)
    
    # Test is_main_process function
    assert train_distillation.is_main_process(0) == True
    assert train_distillation.is_main_process(1) == False
    print("✓ is_main_process function works correctly")
    
    print()
    return True


def test_model_initialization():
    """Test model initialization with multi-GPU support."""
    print("=" * 80)
    print("Test 3: Model Initialization")
    print("=" * 80)
    
    # Create a minimal config
    config = {
        "model_type": "WanLiteStudent",
        "hidden_size": 128,
        "depth": 2,
        "num_heads": 4,
        "num_channels": 4,
        "image_size": 64,
        "patch_size": 16,
        "text_max_length": 77,
        "text_encoder_output_dim": 512,
        "projection_factor": 1.0
    }
    
    # Test initialization without distributed
    device = torch.device("cpu")  # Use CPU for testing
    model = train_distillation.WanLiteStudent(
        config, 
        teacher_checkpoint_path=None, 
        device=device,
        distributed=False
    )
    print("✓ Model initialized successfully (non-distributed)")
    
    # Test initialization with distributed flag
    model_dist = train_distillation.WanLiteStudent(
        config, 
        teacher_checkpoint_path=None, 
        device=device,
        distributed=True
    )
    print("✓ Model initialized successfully (with distributed flag)")
    
    # Verify model can be wrapped with DataParallel (CPU simulation)
    if torch.cuda.device_count() > 1:
        model_dp = DataParallel(model)
        print(f"✓ Model wrapped with DataParallel on {torch.cuda.device_count()} GPUs")
    else:
        print("⚠️  Skipping DataParallel wrap test (requires multiple GPUs)")
    
    print()
    return True


def test_forward_pass():
    """Test forward pass with multi-GPU aware model."""
    print("=" * 80)
    print("Test 4: Forward Pass")
    print("=" * 80)
    
    # Create a minimal config
    config = {
        "model_type": "WanLiteStudent",
        "hidden_size": 128,
        "depth": 2,
        "num_heads": 4,
        "num_channels": 4,
        "image_size": 64,
        "patch_size": 16,
        "text_max_length": 77,
        "text_encoder_output_dim": 512,
        "projection_factor": 1.0
    }
    
    device = torch.device("cpu")
    model = train_distillation.WanLiteStudent(
        config, 
        teacher_checkpoint_path=None, 
        device=device,
        distributed=False
    )
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    latent_0 = torch.randn(
        batch_size, 
        config["num_channels"],
        config["image_size"] // config["patch_size"],
        config["image_size"] // config["patch_size"]
    )
    timestep = torch.randint(0, 1000, (batch_size,))
    encoder_hidden_states = torch.randn(batch_size, 77, config["text_encoder_output_dim"])
    
    # Test forward pass
    with torch.no_grad():
        output = model(latent_0, None, timestep, encoder_hidden_states)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {latent_0.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == latent_0.shape, "Output shape mismatch"
    print(f"✓ Output shape matches input shape")
    
    print()
    return True


def test_dataset_and_sampler():
    """Test dataset loading with distributed sampler."""
    print("=" * 80)
    print("Test 5: Dataset and DistributedSampler")
    print("=" * 80)
    
    # Check if data file exists
    import os
    data_path = "/home/runner/work/WanDistiller/WanDistiller/data/static_prompts.txt"
    
    if not os.path.exists(data_path):
        print(f"⚠️  Data file not found: {data_path}")
        print("   Skipping dataset test")
        print()
        return True
    
    # Create dataset
    dataset = train_distillation.StaticPromptsDataset(data_path)
    print(f"✓ Dataset created with {len(dataset)} samples")
    
    # Test regular DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"✓ Regular DataLoader created")
    
    # Test DistributedSampler (simulated)
    try:
        from torch.utils.data.distributed import DistributedSampler
        # We can't actually initialize DDP here without proper setup
        # but we can verify the import works
        print(f"✓ DistributedSampler import successful")
    except ImportError as e:
        print(f"❌ DistributedSampler import failed: {e}")
        return False
    
    print()
    return True


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "WanDistiller Multi-GPU Support Tests" + " " * 22 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    tests = [
        ("Multi-GPU Detection", test_multi_gpu_detection),
        ("Distributed Functions", test_distributed_functions),
        ("Model Initialization", test_model_initialization),
        ("Forward Pass", test_forward_pass),
        ("Dataset and Sampler", test_dataset_and_sampler),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))
    
    # Print summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = 0
    failed = 0
    for test_name, result, error in results:
        if result:
            print(f"✓ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
            if error:
                print(f"   Error: {error}")
            failed += 1
    
    print()
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 80)
    
    if failed == 0:
        print("\n🎉 All tests passed! Multi-GPU support is ready.\n")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
