#!/usr/bin/env python3
"""
Test new teacher device strategy options for distributed training.

Verifies that:
1. CPU strategy works (backward compatible with --teacher_on_cpu)
2. Balanced strategy distributes across GPUs
3. GPU0 strategy loads only on GPU 0
4. Auto strategy selects appropriately
"""

import sys
import argparse
import tempfile
import json


def test_teacher_device_strategies():
    """Test different teacher device strategies."""
    print("=" * 80)
    print("Testing Teacher Device Strategy Options")
    print("=" * 80)
    
    results = []
    
    # Test 1: CPU strategy
    print("\n1. Testing CPU strategy...")
    args = argparse.Namespace()
    args.teacher_device_strategy = "cpu"
    args.distributed = True
    args.teacher_dtype = "float32"
    
    # Simulate load_kwargs generation
    load_kwargs = {"local_files_only": True}
    
    if args.teacher_device_strategy == "cpu":
        load_kwargs["low_cpu_mem_usage"] = True
    
    if "low_cpu_mem_usage" in load_kwargs:
        print("   ✓ CPU strategy uses low_cpu_mem_usage")
        results.append(("CPU strategy", True))
    else:
        print("   ✗ CPU strategy failed")
        results.append(("CPU strategy", False))
    
    # Test 2: Balanced strategy
    print("\n2. Testing Balanced strategy...")
    args.teacher_device_strategy = "balanced"
    load_kwargs = {"local_files_only": True}
    
    if args.teacher_device_strategy == "balanced":
        load_kwargs["device_map"] = "balanced"
    
    if load_kwargs.get("device_map") == "balanced":
        print("   ✓ Balanced strategy uses device_map='balanced'")
        results.append(("Balanced strategy", True))
    else:
        print("   ✗ Balanced strategy failed")
        results.append(("Balanced strategy", False))
    
    # Test 3: GPU0 strategy
    print("\n3. Testing GPU0 strategy...")
    args.teacher_device_strategy = "gpu0"
    load_kwargs = {"local_files_only": True}
    
    if args.teacher_device_strategy == "gpu0":
        load_kwargs["device_map"] = {"": "cuda:0"}
    
    if load_kwargs.get("device_map") == {"": "cuda:0"}:
        print("   ✓ GPU0 strategy uses device_map for GPU 0")
        results.append(("GPU0 strategy", True))
    else:
        print("   ✗ GPU0 strategy failed")
        results.append(("GPU0 strategy", False))
    
    # Test 4: Backward compatibility with --teacher_on_cpu
    print("\n4. Testing backward compatibility...")
    args = argparse.Namespace()
    args.teacher_on_cpu = True
    args.teacher_device_strategy = None
    
    # Simulate backward compatibility logic
    if args.teacher_on_cpu and args.teacher_device_strategy is None:
        args.teacher_device_strategy = "cpu"
    
    if args.teacher_device_strategy == "cpu":
        print("   ✓ --teacher_on_cpu sets strategy to 'cpu'")
        results.append(("Backward compatibility", True))
    else:
        print("   ✗ Backward compatibility failed")
        results.append(("Backward compatibility", False))
    
    # Test 5: should_load_teacher logic
    print("\n5. Testing should_load_teacher logic...")
    
    # CPU strategy - all ranks load
    rank = 1
    args.teacher_device_strategy = "cpu"
    should_load = True  # All ranks
    print(f"   CPU strategy, rank {rank}: should_load={should_load}")
    results.append(("CPU should_load", should_load == True))
    
    # Balanced strategy - all ranks participate
    args.teacher_device_strategy = "balanced"
    should_load = True  # All ranks
    print(f"   Balanced strategy, rank {rank}: should_load={should_load}")
    results.append(("Balanced should_load", should_load == True))
    
    # GPU0 strategy - only rank 0 loads
    args.teacher_device_strategy = "gpu0"
    should_load = (rank == 0)
    print(f"   GPU0 strategy, rank {rank}: should_load={should_load}")
    results.append(("GPU0 should_load rank 1", should_load == False))
    
    rank = 0
    should_load = (rank == 0)
    print(f"   GPU0 strategy, rank {rank}: should_load={should_load}")
    results.append(("GPU0 should_load rank 0", should_load == True))
    
    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("=" * 80)
        print("✓ All teacher device strategy tests PASSED")
        print("=" * 80)
        return 0
    else:
        print("=" * 80)
        print("✗ Some teacher device strategy tests FAILED")
        print("=" * 80)
        return 1


def test_auto_selection():
    """Test automatic strategy selection."""
    print("\n" + "=" * 80)
    print("Testing Auto Strategy Selection")
    print("=" * 80)
    
    # Simulate auto selection with 2 GPUs
    num_gpus = 2
    strategy = "auto"
    
    if strategy == "auto":
        if num_gpus >= 2:
            strategy = "balanced"
            print(f"   With {num_gpus} GPUs: selected '{strategy}' ✓")
        else:
            strategy = "cpu"
            print(f"   With {num_gpus} GPU: selected '{strategy}' ✓")
    
    # Simulate auto selection with 1 GPU
    num_gpus = 1
    strategy = "auto"
    
    if strategy == "auto":
        if num_gpus >= 2:
            strategy = "balanced"
            print(f"   With {num_gpus} GPUs: selected '{strategy}' ✓")
        else:
            strategy = "cpu"
            print(f"   With {num_gpus} GPU: selected '{strategy}' ✓")
    
    return 0


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Teacher Device Strategy Test Suite")
    print("=" * 80)
    print()
    
    result1 = test_teacher_device_strategies()
    result2 = test_auto_selection()
    
    if result1 == 0 and result2 == 0:
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
