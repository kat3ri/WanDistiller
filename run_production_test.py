#!/usr/bin/env python3
"""
Production Test Script for WanDistiller Training Pipeline

This script runs a real training test with mock data to verify that:
1. The data loading pipeline works correctly
2. The model initialization succeeds
3. The training loop executes without errors
4. The model can be saved successfully

Usage:
    python run_production_test.py [--use-mock-teacher]

Options:
    --use-mock-teacher: Use a mocked teacher model instead of loading real weights
                        (useful for testing when you don't have access to the real model)
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import tempfile
import json

# Import the training module
import train_distillation


def create_mock_teacher_pipeline(device="cpu"):
    """
    Creates a mocked teacher pipeline that simulates the real Diffusers model.
    This allows testing the training loop without needing the actual model weights.
    """
    mock_pipe = MagicMock()
    
    # Mock the transformer
    mock_transformer = MagicMock()
    
    # Create a simple function that returns a tensor when called
    def mock_transformer_forward(*args, **kwargs):
        # Return a tensor matching the expected output shape
        # The shape should match the student's latent spatial dimensions
        sample = kwargs.get('sample')
        if sample is not None:
            batch_size, channels, h, w = sample.shape
            # Return same spatial dimensions as input
            return torch.randn(batch_size, channels, h, w, device=device)
        else:
            # Fallback
            return torch.randn(2, 4, 16, 16, device=device)
    
    mock_transformer.side_effect = mock_transformer_forward
    mock_transformer.eval = MagicMock()
    mock_pipe.transformer = mock_transformer
    
    # Mock encode_prompt to return text embeddings
    def mock_encode_prompt(prompts, device, num_images_per_prompt=1, do_classifier_free_guidance=False):
        batch_size = len(prompts) if isinstance(prompts, list) else 1
        # Return tensor with shape (batch_size, seq_len, text_encoder_output_dim)
        # Standard text sequence length is 77 for most diffusion models
        return (torch.randn(batch_size, 77, 4096, device=device),)
    
    mock_pipe.encode_prompt = mock_encode_prompt
    mock_pipe.to = MagicMock(return_value=mock_pipe)
    
    return mock_pipe


def run_production_test(use_mock_teacher=True, num_epochs=2, batch_size=2):
    """
    Runs a production test of the training pipeline.
    
    Args:
        use_mock_teacher: If True, use a mocked teacher model
        num_epochs: Number of training epochs to run
        batch_size: Batch size for training
    
    Returns:
        bool: True if test passes, False otherwise
    """
    print("=" * 80)
    print("WanDistiller Production Training Test")
    print("=" * 80)
    print()
    
    # Setup paths
    data_path = "data/static_prompts.txt"
    config_path = "config/student_config.json"
    output_dir = tempfile.mkdtemp(prefix="wandistiller_test_")
    
    print(f"Test Configuration:")
    print(f"  - Data path: {data_path}")
    print(f"  - Config path: {config_path}")
    print(f"  - Output dir: {output_dir}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Use mock teacher: {use_mock_teacher}")
    print()
    
    # Check that data file exists
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Data file not found at {data_path}")
        return False
    
    # Count prompts in data file
    with open(data_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"✓ Found {len(prompts)} prompts in data file")
    
    if len(prompts) < 10:
        print(f"⚠️  WARNING: Only {len(prompts)} prompts found. Consider adding more for better testing.")
    
    # Check config file exists
    if not os.path.exists(config_path):
        print(f"❌ ERROR: Config file not found at {config_path}")
        return False
    print(f"✓ Config file found")
    
    # Load and validate config
    try:
        config = train_distillation.load_config(config_path)
        
        # For testing purposes, use a much smaller model to avoid memory issues
        print(f"⚙️  Adjusting model config for testing (reducing size to fit in memory)")
        config["hidden_size"] = 256  # Reduced from 1024
        config["depth"] = 4  # Reduced from 16
        config["num_heads"] = 4  # Reduced from 16
        config["image_size"] = 256  # Reduced from 1024 for faster testing
        
        print(f"✓ Config loaded and adjusted for testing")
        print(f"  - Model type: {config.get('model_type', 'N/A')}")
        print(f"  - Hidden size: {config.get('hidden_size', 'N/A')}")
        print(f"  - Depth: {config.get('depth', 'N/A')}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load config: {e}")
        return False
    
    print()
    print("-" * 80)
    print("Starting Training Test")
    print("-" * 80)
    
    try:
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create dataset
        print("\n1. Initializing dataset...")
        dataset = train_distillation.StaticPromptsDataset(data_path)
        print(f"   ✓ Dataset created with {len(dataset)} samples")
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        print(f"   ✓ DataLoader created (batch_size={batch_size})")
        
        # Initialize student model
        print("\n2. Initializing student model...")
        if use_mock_teacher:
            # Don't pass teacher_checkpoint_path to avoid trying to load real weights
            student_model = train_distillation.WanLiteStudent(
                config, 
                teacher_checkpoint_path=None,
                device=device
            )
        else:
            # This will try to load and project real teacher weights
            # You would need to specify a real teacher_path
            teacher_path = "timbrooks/instruct-wan"
            student_model = train_distillation.WanLiteStudent(
                config,
                teacher_checkpoint_path=teacher_path,
                device=device
            )
        print(f"   ✓ Student model initialized")
        
        # Count parameters
        total_params = sum(p.numel() for p in student_model.parameters())
        trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        print(f"   ✓ Total parameters: {total_params:,}")
        print(f"   ✓ Trainable parameters: {trainable_params:,}")
        
        # Initialize teacher
        print("\n3. Initializing teacher model...")
        if use_mock_teacher:
            print("   Using mocked teacher model...")
            teacher_pipe = create_mock_teacher_pipeline(device=device)
            teacher_model = teacher_pipe.transformer
        else:
            print("   Loading real teacher model...")
            from diffusers import DiffusionPipeline
            teacher_pipe = DiffusionPipeline.from_pretrained("timbrooks/instruct-wan")
            teacher_pipe.to(device)
            teacher_model = teacher_pipe.transformer
            teacher_model.eval()
        print(f"   ✓ Teacher model ready")
        
        # Initialize optimizer
        print("\n4. Initializing optimizer...")
        learning_rate = 1e-5
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
        print(f"   ✓ Optimizer initialized (lr={learning_rate})")
        
        # Training loop
        print(f"\n5. Running training for {num_epochs} epochs...")
        student_model.train()
        global_step = 0
        
        for epoch in range(num_epochs):
            print(f"\n   Epoch {epoch + 1}/{num_epochs}")
            epoch_losses = []
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Get prompts
                    prompts = batch
                    
                    # Encode prompts
                    if use_mock_teacher:
                        text_embeddings = teacher_pipe.encode_prompt(
                            prompts,
                            device=device,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False
                        )[0]
                    else:
                        # Use real text encoder
                        if hasattr(teacher_pipe, 'encode_prompt'):
                            text_embeddings = teacher_pipe.encode_prompt(
                                prompts,
                                device=device,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False
                            )[0]
                        else:
                            # Fallback
                            text_inputs = teacher_pipe.tokenizer(
                                prompts,
                                padding="max_length",
                                max_length=teacher_pipe.tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt"
                            ).to(device)
                            text_embeddings = teacher_pipe.text_encoder(text_inputs.input_ids)[0]
                    
                    # Generate random latents
                    actual_batch_size = len(prompts)
                    latents = torch.randn(
                        actual_batch_size,
                        config["num_channels"],
                        config["image_size"] // config["patch_size"],
                        config["image_size"] // config["patch_size"],
                        device=device
                    )
                    
                    # Generate random timesteps
                    timesteps = torch.randint(0, 1000, (actual_batch_size,), device=device).long()
                    
                    # Teacher forward pass (no gradient)
                    with torch.no_grad():
                        if use_mock_teacher:
                            teacher_output = teacher_model(
                                sample=latents,
                                timestep=timesteps,
                                encoder_hidden_states=text_embeddings
                            )
                        else:
                            teacher_output = teacher_model(
                                sample=latents,
                                timestep=timesteps,
                                encoder_hidden_states=text_embeddings
                            )
                    
                    # Student forward pass
                    student_output = student_model(
                        latent_0=latents,
                        latent_1=None,
                        timestep=timesteps,
                        encoder_hidden_states=text_embeddings
                    )
                    
                    # Calculate loss
                    loss = torch.nn.functional.mse_loss(student_output, teacher_output)
                    
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    global_step += 1
                    
                    # Print progress
                    if batch_idx % max(1, len(dataloader) // 4) == 0:
                        print(f"      Batch {batch_idx + 1}/{len(dataloader)}: Loss = {loss.item():.6f}")
                
                except Exception as e:
                    print(f"   ❌ ERROR in training batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            print(f"   ✓ Epoch {epoch + 1} complete. Average loss: {avg_loss:.6f}")
        
        print(f"\n   ✓ Training completed successfully ({global_step} steps)")
        
        # Test saving
        print("\n6. Testing model save...")
        try:
            student_model.save_pretrained(output_dir)
            print(f"   ✓ Model saved to {output_dir}")
            
            # Check saved files
            saved_files = os.listdir(output_dir)
            print(f"   ✓ Saved files: {', '.join(saved_files)}")
            
            # Verify expected files exist
            if "student_config.json" in saved_files:
                print("   ✓ Config file saved")
            else:
                print("   ⚠️  WARNING: Config file not found in output")
            
            if "diffusion_model.safetensors" in saved_files:
                print("   ✓ Model weights saved")
            else:
                print("   ⚠️  WARNING: Model weights file not found in output")
                
        except Exception as e:
            print(f"   ❌ ERROR saving model: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print()
        print("=" * 80)
        print("✓ Production Test PASSED")
        print("=" * 80)
        print()
        print(f"Test output saved to: {output_dir}")
        print()
        
        return True
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ Production Test FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run production test for WanDistiller training pipeline"
    )
    parser.add_argument(
        "--use-mock-teacher",
        action="store_true",
        default=True,
        help="Use mocked teacher model (default: True)"
    )
    parser.add_argument(
        "--use-real-teacher",
        action="store_true",
        help="Use real teacher model weights (requires model download)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Determine whether to use mock or real teacher
    use_mock = not args.use_real_teacher
    
    # Run the test
    success = run_production_test(
        use_mock_teacher=use_mock,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
