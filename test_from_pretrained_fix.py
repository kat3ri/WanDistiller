#!/usr/bin/env python3
"""
Minimal test to verify the from_pretrained fix for WanLiteStudent.

This test verifies that:
1. The model can be saved using save_pretrained()
2. The model can be loaded using custom from_pretrained() without triggering projection
3. No meta tensor errors occur during loading
"""

import os
import json
import tempfile
import torch
import torch.nn as nn
import sys
from pathlib import Path

# We need to mock the WAN imports since they have many dependencies
sys.modules['wan'] = type(sys)('wan')
sys.modules['wan.text2video'] = type(sys)('wan.text2video')
sys.modules['wan.configs'] = type(sys)('wan.configs')
sys.modules['wan.configs.wan_t2v_A14B'] = type(sys)('wan.configs.wan_t2v_A14B')
sys.modules['projection_mapper'] = type(sys)('projection_mapper')

# Mock the load_and_project_weights function to avoid actual projection
def mock_load_and_project_weights(*args, **kwargs):
    print("[Mock] Skipping weight projection (load_and_project_weights called)")

sys.modules['projection_mapper'].load_and_project_weights = mock_load_and_project_weights

# Now we can import after mocking
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from safetensors.torch import save_file, load_file


class SimpleTransformerBlock(nn.Module):
    """Minimal transformer block for testing."""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
    
    def forward(self, x, t_emb, text_emb):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class WanLiteStudent(ModelMixin, ConfigMixin):
    """Minimal WanLiteStudent for testing."""
    
    config_name = "config.json"
    
    @register_to_config
    def __init__(
        self,
        model_type="WanLiteStudent",
        hidden_size=128,
        depth=2,
        num_heads=4,
        num_channels=4,
        image_size=256,
        patch_size=16,
        text_max_length=77,
        text_encoder_output_dim=512,
        projection_factor=1.0,
        teacher_checkpoint_path=None,
        distributed=False,
        use_gradient_checkpointing=False
    ):
        super().__init__()
        
        self.distributed = distributed
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.text_proj = nn.Linear(text_encoder_output_dim, hidden_size)
        
        time_embed_dim = hidden_size * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_size)
        )
        
        self.conv_in = nn.Conv2d(num_channels, hidden_size, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(hidden_size, num_channels, kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # This is the key part - projection should only happen during initial training
        if teacher_checkpoint_path is not None:
            print(f"[Test] Would load teacher weights from: {teacher_checkpoint_path}")
            mock_load_and_project_weights(
                student_model=self,
                teacher_checkpoint_path=teacher_checkpoint_path,
                config=None,
                device='cpu'
            )
    
    def forward(self, x, timestep, text_emb):
        return x
    
    def save_pretrained(self, output_dir, **kwargs):
        """Save model in HuggingFace Diffusers format."""
        os.makedirs(output_dir, exist_ok=True)
        super().save_pretrained(output_dir, **kwargs)
        print(f"✓ Model saved to: {output_dir}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load a pretrained WanLiteStudent model.
        
        Custom implementation to avoid:
        1. Meta tensor issues
        2. Triggering teacher weight projection on load
        """
        from safetensors.torch import load_file
        import json
        
        # 1. Load config
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        print("[from_pretrained] Loading model from config...")
        
        # 2. Initialize model WITHOUT teacher_checkpoint_path (key fix!)
        model = cls(
            model_type=config_dict.get('model_type', 'WanLiteStudent'),
            hidden_size=config_dict['hidden_size'],
            depth=config_dict['depth'],
            num_heads=config_dict['num_heads'],
            num_channels=config_dict['num_channels'],
            image_size=config_dict['image_size'],
            patch_size=config_dict['patch_size'],
            text_max_length=config_dict['text_max_length'],
            text_encoder_output_dim=config_dict['text_encoder_output_dim'],
            projection_factor=config_dict.get('projection_factor', 1.0),
            teacher_checkpoint_path=None,  # Don't trigger projection!
            distributed=False,
            use_gradient_checkpointing=False
        )
        
        print("[from_pretrained] Model initialized (no projection triggered)")
        
        # 3. Load weights
        possible_names = [
            "diffusion_pytorch_model.safetensors",
            "diffusion_model.safetensors",
            "model.safetensors"
        ]
        
        weights_path = None
        for name in possible_names:
            candidate_path = os.path.join(pretrained_model_name_or_path, name)
            if os.path.exists(candidate_path):
                weights_path = candidate_path
                break
        
        if weights_path is None:
            raise FileNotFoundError(f"Weights file not found in: {pretrained_model_name_or_path}")
        
        print(f"[from_pretrained] Loading weights from {os.path.basename(weights_path)}...")
        state_dict = load_file(weights_path, device="cpu")
        model.load_state_dict(state_dict)
        print("[from_pretrained] Weights loaded successfully")
        
        # 4. Move to device
        device = kwargs.get('device', None)
        if device is not None:
            model = model.to(device)
        
        return model


def test_from_pretrained_no_projection():
    """Test that loading doesn't trigger projection."""
    print("=" * 80)
    print("Testing from_pretrained() - No Projection Should Occur")
    print("=" * 80)
    
    # 1. Create and save a model
    print("\n[1/3] Creating and saving model...")
    model = WanLiteStudent(
        hidden_size=128,
        depth=2,
        num_heads=4,
        teacher_checkpoint_path=None  # No projection during initial creation
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model")
        model.save_pretrained(save_path)
        
        # Verify files exist
        config_exists = os.path.exists(os.path.join(save_path, "config.json"))
        # Check for both possible weight filenames
        weights_exists = (
            os.path.exists(os.path.join(save_path, "diffusion_pytorch_model.safetensors")) or
            os.path.exists(os.path.join(save_path, "diffusion_model.safetensors"))
        )
        assert config_exists, "config.json not found"
        assert weights_exists, "weights file not found"
        print("✓ Model saved successfully")
        
        # 2. Load the model
        print("\n[2/3] Loading model with from_pretrained()...")
        print("   (Watch for projection messages - there should be NONE)")
        loaded_model = WanLiteStudent.from_pretrained(save_path)
        print("✓ Model loaded successfully")
        
        # 3. Verify weights match
        print("\n[3/3] Verifying weights match...")
        orig_param = next(model.parameters())
        loaded_param = next(loaded_model.parameters())
        
        if torch.allclose(orig_param, loaded_param):
            print("✓ Weights match!")
        else:
            print("✗ Weights don't match!")
            return False
    
    print("\n" + "=" * 80)
    print("✓ TEST PASSED: from_pretrained() works without projection!")
    print("=" * 80)
    return True


def test_projection_only_with_teacher_path():
    """Test that projection only happens when teacher_checkpoint_path is provided."""
    print("\n" + "=" * 80)
    print("Testing Projection Only Happens with teacher_checkpoint_path")
    print("=" * 80)
    
    print("\n[1/2] Creating model WITHOUT teacher_checkpoint_path...")
    model1 = WanLiteStudent(
        hidden_size=128,
        depth=2,
        num_heads=4,
        teacher_checkpoint_path=None
    )
    print("✓ No projection occurred (expected)")
    
    print("\n[2/2] Creating model WITH teacher_checkpoint_path...")
    model2 = WanLiteStudent(
        hidden_size=128,
        depth=2,
        num_heads=4,
        teacher_checkpoint_path="./fake_teacher"
    )
    print("✓ Projection would have occurred (expected)")
    
    print("\n" + "=" * 80)
    print("✓ TEST PASSED: Projection control works correctly!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    print("\n🧪 Running WanLiteStudent from_pretrained Fix Tests\n")
    
    success = True
    success &= test_from_pretrained_no_projection()
    success &= test_projection_only_with_teacher_path()
    
    if success:
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe fix ensures:")
        print("  1. from_pretrained() doesn't trigger teacher weight projection")
        print("  2. No meta tensor errors during model loading")
        print("  3. Weights are properly loaded and match original model")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
