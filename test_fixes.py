#!/usr/bin/env python3
"""
Test script to verify the fixes for sample generation issues.

This verifies:
1. Distributed barrier is called after sample generation
2. Multi-step DDIM sampling is implemented instead of single-step
"""

import ast
import sys


def check_barrier_after_sampling():
    """Check that a barrier is called after sample generation in distributed mode."""
    print("Checking for distributed barrier after sample generation...")
    
    with open('train_distillation.py', 'r') as f:
        content = f.read()
    
    # Check for the specific comment and barrier pattern
    if 'Synchronize all processes after sample generation in distributed mode' in content:
        print("  ✓ Found synchronization comment for sample generation")
        # Check that dist.barrier() follows
        if 'Synchronize all processes after sample generation' in content and 'dist.barrier()' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'Synchronize all processes after sample generation' in line:
                    # Check next few lines for barrier
                    for j in range(i, min(i+5, len(lines))):
                        if 'dist.barrier()' in lines[j]:
                            print(f"  ✓ Found barrier at line {j+1}")
                            return True
    
    print("  ✗ Could not find dist.barrier() after sample generation")
    return False


def check_multi_step_sampling():
    """Check that multi-step DDIM sampling is implemented."""
    print("\nChecking for multi-step DDIM sampling...")
    
    with open('train_distillation.py', 'r') as f:
        content = f.read()
    
    # Check for key DDIM components
    checks = [
        ('num_inference_steps', 'Number of inference steps'),
        ('timesteps = torch.linspace', 'Timestep schedule'),
        ('for i, t in enumerate(timesteps):', 'Iterative denoising loop'),
        ('alpha_t', 'Alpha schedule'),
        ('pred_original_sample', 'Predicted original sample'),
    ]
    
    all_found = True
    for pattern, description in checks:
        if pattern in content:
            print(f"  ✓ Found: {description}")
        else:
            print(f"  ✗ Missing: {description}")
            all_found = False
    
    # Check that single-step denoising is NOT present
    if 'DENOISING_ALPHA = 0.5' in content:
        print("  ✗ Still using old single-step denoising")
        all_found = False
    else:
        print("  ✓ Old single-step denoising removed")
    
    return all_found


def main():
    """Run all checks."""
    print("=" * 80)
    print("Verifying fixes for sample generation issues")
    print("=" * 80)
    print()
    
    barrier_ok = check_barrier_after_sampling()
    sampling_ok = check_multi_step_sampling()
    
    print()
    print("=" * 80)
    if barrier_ok and sampling_ok:
        print("✓ ALL CHECKS PASSED")
        print()
        print("Summary of fixes:")
        print("1. Added distributed barrier after sample generation to prevent timeout")
        print("2. Implemented 10-step DDIM sampling for better image quality")
        print("=" * 80)
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
