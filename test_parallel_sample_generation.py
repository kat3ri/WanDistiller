#!/usr/bin/env python3
"""
Test script to verify parallel sample generation across multiple GPUs.

This verifies:
1. generate_and_save_samples function accepts rank and world_size parameters
2. Sample prompts are distributed across ranks
3. All processes participate in sample generation (not just rank 0)
4. Filenames include rank to avoid conflicts
"""

import ast
import sys


def check_function_signature():
    """Check that generate_and_save_samples has rank and world_size parameters."""
    print("Checking function signature for parallel generation...")
    
    with open('train_distillation.py', 'r') as f:
        content = f.read()
    
    # Parse the file to find the function definition
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.args.args:
                if node.name == 'generate_and_save_samples':
                    arg_names = [arg.arg for arg in node.args.args]
                    
                    if 'rank' in arg_names and 'world_size' in arg_names:
                        print(f"  ✓ Function has rank and world_size parameters")
                        print(f"    Parameters: {', '.join(arg_names)}")
                        return True
                    else:
                        print(f"  ✗ Missing rank or world_size parameters")
                        print(f"    Current parameters: {', '.join(arg_names)}")
                        return False
    except Exception as e:
        print(f"  ✗ Error parsing file: {e}")
        return False
    
    print("  ✗ Could not find generate_and_save_samples function")
    return False


def check_prompt_distribution():
    """Check that prompts are distributed across ranks."""
    print("\nChecking for prompt distribution logic...")
    
    with open('train_distillation.py', 'r') as f:
        content = f.read()
    
    checks = [
        ('prompts_per_rank', 'Prompts per rank calculation'),
        ('start_idx', 'Start index for this rank'),
        ('end_idx', 'End index for this rank'),
        ('my_prompts = sample_prompts[start_idx:end_idx]', 'Prompt slicing'),
    ]
    
    all_found = True
    for pattern, description in checks:
        if pattern in content:
            print(f"  ✓ Found: {description}")
        else:
            print(f"  ✗ Missing: {description}")
            all_found = False
    
    return all_found


def check_all_ranks_participate():
    """Check that all ranks participate in sample generation, not just rank 0."""
    print("\nChecking that all ranks participate in sample generation...")
    
    with open('train_distillation.py', 'r') as f:
        lines = f.readlines()
    
    # Look for the sample generation call site
    found_call = False
    all_ranks_participate = False
    
    for i, line in enumerate(lines):
        if 'generate_and_save_samples(' in line:
            found_call = True
            # Check surrounding context (previous 10 lines) for rank check
            context_start = max(0, i - 10)
            context = ''.join(lines[context_start:i+5])
            
            # The old code had: if is_main_process(rank):
            #                       generate_and_save_samples(...)
            # The new code should NOT have that restriction
            if 'if is_main_process(rank):' in context:
                # Check if generate_and_save_samples is indented under this if
                # Count leading spaces
                call_indent = len(line) - len(line.lstrip())
                
                # Look backwards for the if statement
                has_main_process_guard = False
                for j in range(i-1, context_start, -1):
                    if 'if is_main_process(rank):' in lines[j]:
                        if_indent = len(lines[j]) - len(lines[j].lstrip())
                        if call_indent > if_indent:
                            has_main_process_guard = True
                            break
                
                if has_main_process_guard:
                    print(f"  ✗ Sample generation still restricted to main process only")
                    print(f"    Found at line {i+1}")
                    all_ranks_participate = False
                else:
                    print(f"  ✓ All ranks participate in sample generation")
                    all_ranks_participate = True
            else:
                print(f"  ✓ All ranks participate in sample generation")
                all_ranks_participate = True
    
    if not found_call:
        print(f"  ✗ Could not find generate_and_save_samples call")
        return False
    
    return all_ranks_participate


def check_rank_in_filename():
    """Check that rank is included in generated filenames."""
    print("\nChecking for rank in filename...")
    
    with open('train_distillation.py', 'r') as f:
        content = f.read()
    
    # Look for the filename generation
    if 'rank_{rank}' in content or 'rank_' in content and '{rank}' in content:
        print(f"  ✓ Filenames include rank identifier")
        return True
    else:
        print(f"  ✗ Filenames do not include rank identifier")
        return False


def check_world_size_passed():
    """Check that world_size is passed to generate_and_save_samples."""
    print("\nChecking that world_size is passed to function...")
    
    with open('train_distillation.py', 'r') as f:
        content = f.read()
    
    if 'world_size=' in content and 'generate_and_save_samples(' in content:
        # Check if they're in the same call
        lines = content.split('\n')
        in_call = False
        found_world_size = False
        
        for line in lines:
            if 'generate_and_save_samples(' in line:
                in_call = True
            if in_call and 'world_size=' in line:
                found_world_size = True
            if in_call and ')' in line and 'world_size' not in line:
                break
        
        if found_world_size:
            print(f"  ✓ world_size parameter is passed to function")
            return True
    
    print(f"  ✗ world_size parameter not passed to function")
    return False


def main():
    """Run all checks."""
    print("=" * 80)
    print("Verifying parallel sample generation implementation")
    print("=" * 80)
    print()
    
    signature_ok = check_function_signature()
    distribution_ok = check_prompt_distribution()
    all_ranks_ok = check_all_ranks_participate()
    filename_ok = check_rank_in_filename()
    world_size_ok = check_world_size_passed()
    
    print()
    print("=" * 80)
    if signature_ok and distribution_ok and all_ranks_ok and filename_ok and world_size_ok:
        print("✓ ALL CHECKS PASSED")
        print()
        print("Summary of parallel sample generation:")
        print("1. Function signature supports rank and world_size parameters")
        print("2. Prompts are distributed across all available GPUs")
        print("3. All ranks participate in generation (not just rank 0)")
        print("4. Filenames include rank to avoid conflicts")
        print("5. World size is properly passed for distribution calculation")
        print()
        print("Expected speedup: ~Nx faster with N GPUs")
        print("=" * 80)
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
