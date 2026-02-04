#!/usr/bin/env python3
"""
Test script for sample image generation feature.

This script verifies that the sample generation functionality works correctly
with mock components, without requiring the full teacher model.
"""

import torch
import torch.nn as nn
import os
import tempfile
import sys
from unittest.mock import MagicMock


# Mock the WAN imports before importing train_distillation
sys.modules['wan'] = MagicMock()
sys.modules['wan.text2video'] = MagicMock()
sys.modules['wan.configs'] = MagicMock()
sys.modules['wan.configs.wan_t2v_A14B'] = MagicMock()


def create_mock_components():
    """Create mock components for testing sample generation."""
    
    # Mock teacher WAN
    teacher_wan = MagicMock()
    teacher_wan.t5_cpu = False
    
    # Mock text encoder
    def mock_text_encoder(prompts, device):
        batch_size = len(prompts)
        # Return list of embeddings [seq_len, 4096]
        return [torch.randn(77, 4096, device=device) for _ in range(batch_size)]
    
    teacher_text_encoder = MagicMock(side_effect=mock_text_encoder)
    
    # Mock VAE
    def mock_vae_decode(latents_list):
        # Return list of videos [C, F, H, W]
        results = []
        for latent in latents_list:
            C, F, H, W = latent.shape
            # Return random video tensor in [-1, 1] range
            video = torch.randn(C, F, H, W) * 2 - 1
            results.append(video)
        return results
    
    teacher_vae = MagicMock()
    teacher_vae.decode = MagicMock(side_effect=mock_vae_decode)
    
    return teacher_wan, teacher_text_encoder, teacher_vae


def create_mock_student():
    """Create a simple mock student model."""
    
    class MockStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 3, padding=1)
        
        def forward(self, latent_0, latent_1, timestep, encoder_hidden_states):
            # Simple pass through
            return self.conv(latent_0)
        
        def train(self):
            pass
        
        def eval(self):
            pass
    
    return MockStudent()


def test_sample_generation():
    """Test the sample generation function."""
    
    print("=" * 80)
    print("Testing Sample Image Generation Feature")
    print("=" * 80)
    print()
    
    # Import the function
    sys.path.insert(0, os.path.dirname(__file__))
    from train_distillation import generate_and_save_samples
    
    # Create mock components
    print("1. Creating mock components...")
    teacher_wan, teacher_text_encoder, teacher_vae = create_mock_components()
    student_model = create_mock_student()
    print("✓ Mock components created")
    print()
    
    # Setup test configuration
    print("2. Setting up test configuration...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_device = device
    teacher_dtype = torch.float32
    
    student_config = {
        "num_channels": 4,
        "image_size": 256,
        "patch_size": 16,
    }
    
    sample_prompts = ["Test prompt 1", "Test prompt 2"]
    print(f"✓ Using device: {device}")
    print(f"✓ Sample prompts: {sample_prompts}")
    print()
    
    # Create temporary directory for samples
    print("3. Testing sample generation...")
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = os.path.join(tmpdir, "samples")
        
        try:
            # Call the generation function
            generate_and_save_samples(
                student_model=student_model,
                teacher_text_encoder=teacher_text_encoder,
                teacher_vae=teacher_vae,
                teacher_wan=teacher_wan,
                sample_prompts=sample_prompts,
                sample_dir=sample_dir,
                epoch=1,
                device=device,
                teacher_device=teacher_device,
                teacher_dtype=teacher_dtype,
                student_config=student_config,
                proj_layer=None
            )
            
            # Check that images were created
            expected_files = [
                "epoch_0001_sample_00.png",
                "epoch_0001_sample_01.png"
            ]
            
            print()
            print("4. Verifying generated files...")
            for filename in expected_files:
                filepath = os.path.join(sample_dir, filename)
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    print(f"✓ Found: {filename} ({file_size} bytes)")
                else:
                    print(f"✗ Missing: {filename}")
                    return False
            
            print()
            print("=" * 80)
            print("✓ ALL TESTS PASSED")
            print("=" * 80)
            return True
            
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = test_sample_generation()
    sys.exit(0 if success else 1)
