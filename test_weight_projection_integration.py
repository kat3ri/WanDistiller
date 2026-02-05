#!/usr/bin/env python3
"""
Integration test for weight projection from teacher to student model.

This test validates that weight projection works end-to-end with a mock teacher model.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from projection_mapper import load_and_project_weights


def create_mock_teacher_state_dict():
    """
    Create a smaller mock teacher state dict for testing.
    
    Teacher architecture (simplified):
    - 4 transformer blocks (instead of 40)
    - 512 hidden dimensions (instead of 5120)
    - Conv3D layers (with temporal dimension)
    """
    state_dict = {}
    
    # Text projection: 256 → 512
    state_dict['text_proj.weight'] = torch.randn(512, 256)
    state_dict['text_proj.bias'] = torch.randn(512)
    
    # Time embedding: 512 → 2048 → 512
    state_dict['time_embed.0.weight'] = torch.randn(2048, 512)
    state_dict['time_embed.0.bias'] = torch.randn(2048)
    state_dict['time_embed.2.weight'] = torch.randn(512, 2048)
    state_dict['time_embed.2.bias'] = torch.randn(512)
    
    # Conv3D input: (out=512, in=4, T=3, H=3, W=3)
    state_dict['conv_in.weight'] = torch.randn(512, 4, 3, 3, 3)
    state_dict['conv_in.bias'] = torch.randn(512)
    
    # 4 Transformer blocks (instead of 40)
    for i in range(4):
        # Layer norms
        state_dict[f'blocks.{i}.norm1.weight'] = torch.randn(512)
        state_dict[f'blocks.{i}.norm1.bias'] = torch.randn(512)
        state_dict[f'blocks.{i}.norm2.weight'] = torch.randn(512)
        state_dict[f'blocks.{i}.norm2.bias'] = torch.randn(512)
        
        # Attention
        state_dict[f'blocks.{i}.attn.in_proj_weight'] = torch.randn(3 * 512, 512)
        state_dict[f'blocks.{i}.attn.in_proj_bias'] = torch.randn(3 * 512)
        state_dict[f'blocks.{i}.attn.out_proj.weight'] = torch.randn(512, 512)
        state_dict[f'blocks.{i}.attn.out_proj.bias'] = torch.randn(512)
        
        # MLP: 512 → 2048 → 512
        state_dict[f'blocks.{i}.mlp.0.weight'] = torch.randn(2048, 512)
        state_dict[f'blocks.{i}.mlp.0.bias'] = torch.randn(2048)
        state_dict[f'blocks.{i}.mlp.2.weight'] = torch.randn(512, 2048)
        state_dict[f'blocks.{i}.mlp.2.bias'] = torch.randn(512)
    
    # Conv3D output: (out=4, in=512, T=3, H=3, W=3)
    state_dict['conv_out.weight'] = torch.randn(4, 512, 3, 3, 3)
    state_dict['conv_out.bias'] = torch.randn(4)
    
    return state_dict


def create_student_model():
    """
    Create a smaller student model for testing.
    
    Student architecture:
    - 2 transformer blocks (instead of 16)
    - 128 hidden dimensions (instead of 1024)
    - 8 attention heads
    - Conv2D layers (2D only)
    """
    class WanTransformerBlock(nn.Module):
        def __init__(self, hidden_size, num_heads):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden_size)
            self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, 4 * hidden_size),
                nn.SiLU(),
                nn.Linear(4 * hidden_size, hidden_size)
            )
        
        def forward(self, x):
            attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
            x = x + attn_output
            x = x + self.mlp(self.norm2(x))
            return x
    
    class StudentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_proj = nn.Linear(256, 128)
            self.time_embed = nn.Sequential(
                nn.Linear(128, 512),
                nn.SiLU(),
                nn.Linear(512, 128)
            )
            self.conv_in = nn.Conv2d(4, 128, kernel_size=3, padding=1)
            self.blocks = nn.ModuleList([
                WanTransformerBlock(128, 8) for _ in range(2)
            ])
            self.conv_out = nn.Conv2d(128, 4, kernel_size=3, padding=1)
        
        def forward(self, x):
            # Return input unchanged (intentionally simple for testing purposes)
            return x
    
    return StudentModel()


def test_weight_projection_integration():
    """
    Integration test for weight projection.
    
    Tests:
    1. Loading mock teacher state_dict
    2. Projecting weights to student model
    3. Verifying weights were transferred
    4. Checking projection statistics
    """
    print("=" * 80)
    print("Weight Projection Integration Test")
    print("=" * 80)
    print()
    
    # Step 1: Create mock teacher state dict
    print("[Test] Creating mock teacher state dict...")
    teacher_state_dict = create_mock_teacher_state_dict()
    print(f"[Test] Teacher state dict: {len(teacher_state_dict)} keys")
    print(f"[Test] Sample teacher shapes:")
    print(f"  - text_proj.weight: {teacher_state_dict['text_proj.weight'].shape}")
    print(f"  - conv_in.weight: {teacher_state_dict['conv_in.weight'].shape}")
    print(f"  - blocks.0.attn.in_proj_weight: {teacher_state_dict['blocks.0.attn.in_proj_weight'].shape}")
    print()
    
    # Step 2: Create student model
    print("[Test] Creating student model...")
    student_model = create_student_model()
    
    # Save original weights to check they change
    original_text_proj = student_model.text_proj.weight.clone()
    original_conv_in = student_model.conv_in.weight.clone()
    original_block0_norm = student_model.blocks[0].norm1.weight.clone()
    
    print(f"[Test] Student model structure:")
    print(f"  - text_proj: {student_model.text_proj.weight.shape}")
    print(f"  - conv_in: {student_model.conv_in.weight.shape}")
    print(f"  - blocks: {len(student_model.blocks)} layers")
    print()
    
    # Step 3: Define config
    class MockConfig:
        hidden_size = 128
        depth = 2
        num_heads = 8
    
    config = MockConfig()
    
    # Step 4: Apply weight projection
    print("[Test] Applying weight projection...")
    print()
    
    student_model = load_and_project_weights(
        student_model=student_model,
        teacher_checkpoint_path=teacher_state_dict,
        config=config,
        device='cpu',
        projection_method='truncate'
    )
    
    print()
    print("[Test] Validating projection results...")
    print()
    
    # Step 5: Verify weights changed
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Text projection weight changed
    if not torch.equal(student_model.text_proj.weight, original_text_proj):
        print("✓ Test 1 PASSED: text_proj weights changed")
        tests_passed += 1
    else:
        print("✗ Test 1 FAILED: text_proj weights unchanged")
        tests_failed += 1
    
    # Test 2: Text projection has correct shape after projection
    if student_model.text_proj.weight.shape == (128, 256):
        print("✓ Test 2 PASSED: text_proj shape correct (128, 256)")
        tests_passed += 1
    else:
        print(f"✗ Test 2 FAILED: text_proj shape incorrect {student_model.text_proj.weight.shape}")
        tests_failed += 1
    
    # Test 3: Conv weight changed (and converted from Conv3D to Conv2D)
    if not torch.equal(student_model.conv_in.weight, original_conv_in):
        print("✓ Test 3 PASSED: conv_in weights changed")
        tests_passed += 1
    else:
        print("✗ Test 3 FAILED: conv_in weights unchanged")
        tests_failed += 1
    
    # Test 4: Conv has correct 2D shape
    if student_model.conv_in.weight.dim() == 4:
        print(f"✓ Test 4 PASSED: conv_in is 2D {student_model.conv_in.weight.shape}")
        tests_passed += 1
    else:
        print(f"✗ Test 4 FAILED: conv_in is not 2D {student_model.conv_in.weight.shape}")
        tests_failed += 1
    
    # Test 5: Block weights changed
    if not torch.equal(student_model.blocks[0].norm1.weight, original_block0_norm):
        print("✓ Test 5 PASSED: block weights changed")
        tests_passed += 1
    else:
        print("✗ Test 5 FAILED: block weights unchanged")
        tests_failed += 1
    
    # Test 6: Verify weight magnitudes are reasonable
    text_proj_std = student_model.text_proj.weight.std()
    if 0.01 < text_proj_std < 10.0:
        print(f"✓ Test 6 PASSED: text_proj weight std is reasonable ({text_proj_std:.4f})")
        tests_passed += 1
    else:
        print(f"✗ Test 6 FAILED: text_proj weight std is unreasonable ({text_proj_std:.4f})")
        tests_failed += 1
    
    # Test 7: Verify all blocks have weights (not all zeros)
    all_blocks_have_weights = True
    for i, block in enumerate(student_model.blocks):
        if block.norm1.weight.abs().max() < 1e-6:
            all_blocks_have_weights = False
            print(f"✗ Block {i} has zero weights")
            break
    
    if all_blocks_have_weights:
        print(f"✓ Test 7 PASSED: All {len(student_model.blocks)} blocks have non-zero weights")
        tests_passed += 1
    else:
        print(f"✗ Test 7 FAILED: Some blocks have zero weights")
        tests_failed += 1
    
    # Test 8: Test forward pass works
    try:
        # Create dummy input
        batch_size = 2
        seq_len = 8
        hidden_size = 128
        dummy_input = torch.randn(batch_size, seq_len, hidden_size)
        
        # Test first block
        output = student_model.blocks[0](dummy_input)
        
        if output.shape == dummy_input.shape:
            print("✓ Test 8 PASSED: Forward pass works correctly")
            tests_passed += 1
        else:
            print(f"✗ Test 8 FAILED: Output shape mismatch {output.shape} vs {dummy_input.shape}")
            tests_failed += 1
    except Exception as e:
        print(f"✗ Test 8 FAILED: Forward pass error: {e}")
        tests_failed += 1
    
    # Final summary
    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests Passed: {tests_passed}/8")
    print(f"Tests Failed: {tests_failed}/8")
    
    if tests_failed == 0:
        print()
        print("✅ ALL TESTS PASSED - Weight projection is working correctly!")
        print()
        return True
    else:
        print()
        print("❌ SOME TESTS FAILED - Check the output above")
        print()
        return False


if __name__ == '__main__':
    success = test_weight_projection_integration()
    sys.exit(0 if success else 1)
