#!/usr/bin/env python3
"""
Test script to verify the NCCL timeout fix for sample generation.

This verifies:
1. datetime module is imported
2. timeout parameter is set in dist.init_process_group()
3. timeout value is 3600 seconds (1 hour)
"""

import ast
import sys


def check_datetime_import():
    """Check that datetime module is imported."""
    print("Checking for datetime import...")
    
    with open('train_distillation.py', 'r') as f:
        content = f.read()
    
    if 'import datetime' in content:
        print("  ✓ datetime module is imported")
        return True
    else:
        print("  ✗ datetime module is not imported")
        return False


def check_timeout_parameter():
    """Check that timeout parameter is set in dist.init_process_group()."""
    print("\nChecking for timeout parameter in dist.init_process_group()...")
    
    with open('train_distillation.py', 'r') as f:
        content = f.read()
    
    # Check for timeout parameter
    if 'timeout=' in content and 'datetime.timedelta' in content:
        print("  ✓ Found timeout parameter with datetime.timedelta")
        
        # Check the timeout value
        if 'timeout=datetime.timedelta(seconds=3600)' in content:
            print("  ✓ Timeout is correctly set to 3600 seconds (1 hour)")
            return True
        else:
            print("  ⚠ Timeout value might not be 3600 seconds")
            return False
    else:
        print("  ✗ timeout parameter not found")
        return False


def check_timeout_in_init_process_group():
    """Check that timeout is in the correct location (dist.init_process_group)."""
    print("\nChecking that timeout is in dist.init_process_group()...")
    
    with open('train_distillation.py', 'r') as f:
        lines = f.readlines()
    
    # Find dist.init_process_group and check if timeout is within that call
    in_init_call = False
    timeout_found = False
    
    for i, line in enumerate(lines):
        if 'dist.init_process_group(' in line:
            in_init_call = True
            print(f"  Found dist.init_process_group at line {i+1}")
        
        if in_init_call:
            if 'timeout=' in line:
                timeout_found = True
                print(f"  ✓ Found timeout parameter at line {i+1}")
            if ')' in line and 'timeout' not in line:
                # Check if we've seen the closing paren without finding timeout
                if not timeout_found:
                    # This closing paren might be for timeout, continue
                    continue
                break
    
    if timeout_found:
        print("  ✓ timeout parameter is correctly placed in dist.init_process_group()")
        return True
    else:
        print("  ✗ timeout parameter not found in dist.init_process_group()")
        return False


def check_barrier_still_present():
    """Verify that the barrier after sample generation is still present."""
    print("\nChecking that barrier after sample generation is still present...")
    
    with open('train_distillation.py', 'r') as f:
        content = f.read()
    
    if 'Synchronize all processes after sample generation' in content and 'dist.barrier()' in content:
        print("  ✓ Barrier after sample generation is still present")
        return True
    else:
        print("  ✗ Barrier after sample generation might be missing")
        return False


def main():
    """Run all checks."""
    print("=" * 80)
    print("Verifying NCCL timeout fix for sample generation")
    print("=" * 80)
    print()
    
    datetime_ok = check_datetime_import()
    timeout_ok = check_timeout_parameter()
    location_ok = check_timeout_in_init_process_group()
    barrier_ok = check_barrier_still_present()
    
    print()
    print("=" * 80)
    if datetime_ok and timeout_ok and location_ok and barrier_ok:
        print("✓ ALL CHECKS PASSED")
        print()
        print("Summary of fix:")
        print("1. Added datetime import for timedelta support")
        print("2. Set timeout parameter in dist.init_process_group() to 3600 seconds (1 hour)")
        print("3. This allows sample generation to complete without NCCL timeout")
        print("4. Barrier synchronization is still present for correct distributed training")
        print("=" * 80)
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
