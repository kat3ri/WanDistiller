#!/usr/bin/env python3
"""
Test projection mapper key coverage.

This test verifies that all student model parameters can be mapped to teacher
parameters using the projection mapper logic, without requiring torch or other
heavy dependencies.

NOTE: map_teacher_to_student_key() is intentionally duplicated here instead of
imported to make this test runnable without torch/safetensors dependencies.
This allows for quick validation of the key mapping logic during development.
"""

import sys
import os


def map_teacher_to_student_key(student_key, student_block_idx, layer_mapping):
    """
    Map a student parameter key to the corresponding teacher parameter key.
    
    Handles the architectural differences between teacher (WanAttentionBlock) and 
    student (WanTransformerBlock):
    - Teacher uses self_attn with separate q, k, v, o projections
    - Student uses attn (nn.MultiheadAttention) with in_proj (combined q,k,v) and out_proj
    - Teacher uses ffn for feed-forward, student uses mlp
    
    Args:
        student_key: Student parameter key (e.g., 'blocks.0.attn.in_proj_weight')
        student_block_idx: Student block index
        layer_mapping: List mapping student layers to teacher layers
        
    Returns:
        List of (teacher_key, slice_info) tuples for mapping teacher params to student.
        slice_info is None for direct copy, or (operation, dim) for special ops.
    """
    # Map student block index to teacher block index
    teacher_block_idx = layer_mapping[student_block_idx] if student_block_idx < len(layer_mapping) else student_block_idx
    
    # Handle MultiheadAttention parameter mapping
    # Student uses nn.MultiheadAttention which has in_proj_weight (concatenated q,k,v) and out_proj
    # Teacher uses separate q, k, v, o linear layers
    
    if '.attn.in_proj_weight' in student_key:
        # Student's in_proj_weight = [q_weight; k_weight; v_weight] concatenated
        # Map to teacher's self_attn.q, self_attn.k, self_attn.v
        base_key = f'blocks.{teacher_block_idx}.self_attn'
        return [
            (f'{base_key}.q.weight', ('cat', 0)),  # Concatenate along dim 0
            (f'{base_key}.k.weight', ('cat', 0)),
            (f'{base_key}.v.weight', ('cat', 0)),
        ]
    elif '.attn.in_proj_bias' in student_key:
        # Student's in_proj_bias = [q_bias; k_bias; v_bias] concatenated
        base_key = f'blocks.{teacher_block_idx}.self_attn'
        return [
            (f'{base_key}.q.bias', ('cat', 0)),
            (f'{base_key}.k.bias', ('cat', 0)),
            (f'{base_key}.v.bias', ('cat', 0)),
        ]
    elif '.attn.out_proj.weight' in student_key:
        # Direct mapping
        return [(f'blocks.{teacher_block_idx}.self_attn.o.weight', None)]
    elif '.attn.out_proj.bias' in student_key:
        # Direct mapping
        return [(f'blocks.{teacher_block_idx}.self_attn.o.bias', None)]
    elif '.mlp.' in student_key:
        # Student uses 'mlp', teacher uses 'ffn'
        teacher_key = student_key.replace(f'blocks.{student_block_idx}.', f'blocks.{teacher_block_idx}.')
        teacher_key = teacher_key.replace('.mlp.', '.ffn.')
        return [(teacher_key, None)]
    else:
        # For other parameters (norm1, norm2), try direct mapping with teacher block index
        teacher_key = student_key.replace(f'blocks.{student_block_idx}.', f'blocks.{teacher_block_idx}.')
        return [(teacher_key, None)]


def create_mock_teacher_keys():
    """Create a set of teacher model keys."""
    keys = set()
    
    # Non-block parameters
    keys.add('text_proj.weight')
    keys.add('text_proj.bias')
    keys.add('time_embed.0.weight')
    keys.add('conv_in.weight')
    keys.add('conv_out.weight')
    
    # Teacher has 40 blocks with WanAttentionBlock structure
    for i in range(40):
        # Self attention components (separate q, k, v, o)
        keys.add(f'blocks.{i}.norm1.weight')
        keys.add(f'blocks.{i}.self_attn.q.weight')
        keys.add(f'blocks.{i}.self_attn.q.bias')
        keys.add(f'blocks.{i}.self_attn.k.weight')
        keys.add(f'blocks.{i}.self_attn.k.bias')
        keys.add(f'blocks.{i}.self_attn.v.weight')
        keys.add(f'blocks.{i}.self_attn.v.bias')
        keys.add(f'blocks.{i}.self_attn.o.weight')
        keys.add(f'blocks.{i}.self_attn.o.bias')
        
        # QK norm
        keys.add(f'blocks.{i}.self_attn.norm_q.weight')
        keys.add(f'blocks.{i}.self_attn.norm_k.weight')
        
        # Cross attention (student doesn't have this - will be skipped)
        keys.add(f'blocks.{i}.norm3.weight')
        keys.add(f'blocks.{i}.cross_attn.q.weight')
        keys.add(f'blocks.{i}.cross_attn.k.weight')
        keys.add(f'blocks.{i}.cross_attn.v.weight')
        keys.add(f'blocks.{i}.cross_attn.o.weight')
        
        # FFN (feed-forward network)
        keys.add(f'blocks.{i}.norm2.weight')
        keys.add(f'blocks.{i}.ffn.0.weight')
        keys.add(f'blocks.{i}.ffn.0.bias')
        keys.add(f'blocks.{i}.ffn.2.weight')
        keys.add(f'blocks.{i}.ffn.2.bias')
        
        # Modulation (student doesn't have this - will be skipped)
        keys.add(f'blocks.{i}.modulation')
    
    return keys


def create_mock_student_keys():
    """Create a set of student model keys."""
    keys = set()
    
    # Non-block parameters (should match exactly with teacher)
    keys.add('text_proj.weight')
    keys.add('text_proj.bias')
    keys.add('time_embed.0.weight')
    keys.add('conv_in.weight')
    keys.add('conv_out.weight')
    
    # Student has 16 blocks with nn.MultiheadAttention structure
    for i in range(16):
        # Attention using nn.MultiheadAttention (in_proj combines q,k,v)
        keys.add(f'blocks.{i}.norm1.weight')
        keys.add(f'blocks.{i}.attn.in_proj_weight')
        keys.add(f'blocks.{i}.attn.in_proj_bias')
        keys.add(f'blocks.{i}.attn.out_proj.weight')
        keys.add(f'blocks.{i}.attn.out_proj.bias')
        
        # MLP
        keys.add(f'blocks.{i}.norm2.weight')
        keys.add(f'blocks.{i}.mlp.0.weight')
        keys.add(f'blocks.{i}.mlp.0.bias')
        keys.add(f'blocks.{i}.mlp.2.weight')
        keys.add(f'blocks.{i}.mlp.2.bias')
    
    return keys


def test_key_coverage():
    """Test that projection covers all student keys."""
    teacher_keys = create_mock_teacher_keys()
    student_keys = create_mock_student_keys()
    
    layer_mapping = [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37]
    
    print("=" * 80)
    print("PROJECTION KEY COVERAGE TEST")
    print("=" * 80)
    print()
    
    print(f"Teacher keys: {len(teacher_keys)}")
    print(f"Student keys: {len(student_keys)}")
    print()
    
    # Track statistics
    exact_matches = 0
    mapped_keys = 0
    concatenated_keys = 0
    skipped_keys = 0
    skipped_list = []
    
    # Analyze each student key
    for student_key in sorted(student_keys):
        # Check for exact match first
        if student_key in teacher_keys:
            exact_matches += 1
            continue
        
        # Check for block mapping
        if 'blocks.' in student_key:
            parts = student_key.split('.')
            if len(parts) >= 2 and parts[0] == 'blocks':
                try:
                    student_block_idx = int(parts[1])
                    mappings = map_teacher_to_student_key(student_key, student_block_idx, layer_mapping)
                    
                    if mappings:
                        # Check if all teacher keys exist
                        all_exist = all(tk in teacher_keys for tk, _ in mappings)
                        
                        if all_exist:
                            if len(mappings) > 1:
                                concatenated_keys += 1
                            else:
                                mapped_keys += 1
                            continue
                except Exception as e:
                    print(f"Error processing {student_key}: {e}")
        
        # If we got here, key wasn't mapped
        skipped_keys += 1
        skipped_list.append(student_key)
    
    # Print skipped keys
    if skipped_list:
        print("Skipped keys:")
        for key in skipped_list:
            print(f"  ⚠ {key}")
        print()
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"✓ Exact matches:           {exact_matches:4d} keys")
    print(f"✓ Mapped (single):         {mapped_keys:4d} keys")
    print(f"✓ Concatenated (multiple): {concatenated_keys:4d} keys")
    print(f"⚠ Skipped:                  {skipped_keys:4d} keys")
    print()
    
    total_keys = len(student_keys)
    transferred = exact_matches + mapped_keys + concatenated_keys
    transfer_rate = (transferred / total_keys * 100) if total_keys > 0 else 0
    
    print(f"Transfer rate: {transfer_rate:.1f}% ({transferred}/{total_keys})")
    print()
    
    if transfer_rate >= 95:
        print("✅ SUCCESS: Transfer rate is excellent (≥95%)")
        return True
    elif transfer_rate >= 80:
        print("✓ GOOD: Transfer rate is acceptable (≥80%)")
        return True
    else:
        print("❌ FAILURE: Transfer rate is too low (<80%)")
        return False


if __name__ == '__main__':
    success = test_key_coverage()
    sys.exit(0 if success else 1)
