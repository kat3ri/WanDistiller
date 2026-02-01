"""
Mock Teacher Model Implementation

This module provides a mock implementation of the Wan 2.2 T2V teacher model
that can be used for testing when the real model is not available.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class MockTransformer(nn.Module):
    """Mock transformer that mimics the Wan 2.2 T2V transformer behavior."""
    
    def __init__(self, hidden_size=1024, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.device = device
        
        # Simple layers to make it more realistic
        self.time_proj = nn.Linear(hidden_size, hidden_size)
        self.text_proj = nn.Linear(4096, hidden_size)
        self.conv = nn.Conv2d(4, hidden_size, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(hidden_size, 4, kernel_size=3, padding=1)
        
        self.to(device)
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs
    ):
        """Forward pass that mimics real transformer output."""
        batch_size, channels, h, w = sample.shape
        
        # Simple processing to make output related to input
        x = self.conv(sample)
        
        # Add some conditioning (simplified)
        x = x + self.text_proj(encoder_hidden_states.mean(dim=1))[:, :, None, None]
        
        # Output
        output = self.conv_out(x)
        
        # Add some noise based on timestep to simulate diffusion behavior
        noise_scale = timestep.float().view(-1, 1, 1, 1) / 1000.0
        output = output + torch.randn_like(output) * noise_scale * 0.1
        
        # Return in the format that Diffusers models typically use
        class TransformerOutput:
            def __init__(self, sample):
                self.sample = sample
        
        return TransformerOutput(output)
    
    def eval(self):
        """Set to eval mode."""
        return super().eval()


class MockTextEncoder(nn.Module):
    """Mock text encoder for encoding prompts."""
    
    def __init__(self, output_dim=4096, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = output_dim
        self.device = device
        self.embedding = nn.Embedding(50000, output_dim)
        self.to(device)
    
    def forward(self, input_ids):
        return self.embedding(input_ids)


class MockTokenizer:
    """Mock tokenizer for processing text prompts."""
    
    def __init__(self, model_max_length=77):
        self.model_max_length = model_max_length
    
    def __call__(
        self,
        prompts,
        padding="max_length",
        max_length=None,
        truncation=True,
        return_tensors="pt"
    ):
        """Tokenize prompts and return mock input."""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        batch_size = len(prompts)
        max_len = max_length or self.model_max_length
        
        # Create random token IDs (simulating tokenization)
        input_ids = torch.randint(0, 50000, (batch_size, max_len))
        
        class TokenizerOutput:
            def __init__(self, input_ids):
                self.input_ids = input_ids
            
            def to(self, device):
                self.input_ids = self.input_ids.to(device)
                return self
        
        return TokenizerOutput(input_ids)


class MockDiffusionPipeline:
    """Mock diffusion pipeline that mimics DiffusionPipeline.from_pretrained behavior."""
    
    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.transformer = MockTransformer(device=device)
        self.text_encoder = MockTextEncoder(device=device)
        self.tokenizer = MockTokenizer()
    
    def to(self, device):
        """Move pipeline to device."""
        self.device = device
        self.transformer.to(device)
        self.text_encoder.to(device)
        return self
    
    def encode_prompt(
        self,
        prompts,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        **kwargs
    ):
        """Encode prompts to text embeddings."""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        batch_size = len(prompts)
        
        # Tokenize
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Encode
        text_embeddings = self.text_encoder(text_inputs.input_ids)
        
        return (text_embeddings,)
    
    @staticmethod
    def from_pretrained(model_id, **kwargs):
        """Mock the from_pretrained method."""
        print(f"Creating mock teacher pipeline for: {model_id}")
        print("Note: Using simulated model as real model is not available")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return MockDiffusionPipeline(device=device)


def create_mock_teacher_pipeline(teacher_path: str, device=None):
    """
    Create a mock teacher pipeline that simulates the real Wan 2.2 model.
    
    Args:
        teacher_path: Path/ID of the teacher model (ignored in mock)
        device: Device to place the model on
    
    Returns:
        MockDiffusionPipeline instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Mock Teacher] Creating mock teacher model (real model not available)")
    print(f"[Mock Teacher] Model ID: {teacher_path}")
    print(f"[Mock Teacher] Device: {device}")
    
    pipeline = MockDiffusionPipeline(device=device)
    return pipeline
