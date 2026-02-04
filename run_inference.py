#!/usr/bin/env python3
"""
Command-Line Interface for running inference with a trained WanLiteStudent model.

This script loads a saved student model, uses the teacher's VAE and text encoder,
and generates an image from a given text prompt.

Example Usage:
  python run_inference.py \
    --model_path "./outputs/wan_t2i" \
    --teacher_path "./Wan2.2-T2V-A14B" \
    --prompt "A beautiful mountain landscape at sunset" \
    --output_path "inference_result.png"
"""

import argparse
import torch
from PIL import Image
import os
import sys
import warnings

# Add the directory containing train_distillation to the Python path
# This is necessary to import WanLiteStudent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from train_distillation import WanLiteStudent
    from wan.text2video import WanT2V
    from wan.configs.wan_t2v_A14B import t2v_A14B
    from easydict import EasyDict
except ImportError as e:
    print("="*80, file=sys.stderr)
    print("ERROR: Failed to import necessary modules.", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    print("\nPlease ensure you are running this script from the 'WanDistiller' directory", file=sys.stderr)
    print("and that all dependencies from 'requirements.txt' are installed.", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained WanLiteStudent model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved student model directory (e.g., ./outputs/wan_t2i).")
    parser.add_argument("--teacher_path", type=str, required=True, help="Path to the original WAN teacher model directory (for VAE and text encoder).")
    parser.add_argument("--prompt", type=str, required=True, help="The text prompt for image generation.")
    parser.add_argument("--output_path", type=str, default="inference_result.png", help="Path to save the generated image.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of DDIM inference steps.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on ('cuda' or 'cpu').")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the trained student model
    print(f"Loading student model from {args.model_path}...")
    try:
        student_model = WanLiteStudent.from_pretrained(args.model_path)
        student_model.to(device)
        student_model.eval()
        print("✓ Student model loaded.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load student model from {args.model_path}", file=sys.stderr)
        print(f"   Please ensure the path is correct and contains 'config.json' and 'diffusion_model.safetensors'.", file=sys.stderr)
        print(f"   Error details: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Load teacher components (VAE and Text Encoder)
    print(f"Loading teacher components from {args.teacher_path}...")
    try:
        teacher_config = EasyDict(t2v_A14B)
        # Suppress warnings about tied weights in T5/UMT5 models
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*were not used when initializing.*")
            warnings.filterwarnings("ignore", message=".*were newly initialized.*")
            teacher_wan = WanT2V(
                config=teacher_config,
                checkpoint_dir=args.teacher_path,
                device_id=0 if device.type == 'cuda' else -1,
                t5_cpu=True,  # Keep T5 on CPU to save VRAM
                init_on_cpu=True
            )
        teacher_text_encoder = teacher_wan.text_encoder
        teacher_vae = teacher_wan.vae
        print("✓ Teacher components loaded.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load teacher components from {args.teacher_path}", file=sys.stderr)
        print(f"   Please ensure the path is correct and the model is downloaded.", file=sys.stderr)
        print(f"   Error details: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Inference loop
    with torch.no_grad():
        # Encode prompt
        print(f"Encoding prompt: '{args.prompt}'")
        text_embeddings_list = teacher_text_encoder([args.prompt], torch.device('cpu'))
        text_embeddings_list = [t.to(device) for t in text_embeddings_list]
        
        # Stack and pad to create a batch tensor
        max_len = max(t.shape[0] for t in text_embeddings_list)
        text_embeddings = torch.zeros(1, max_len, text_embeddings_list[0].shape[1], dtype=text_embeddings_list[0].dtype, device=device)
        text_embeddings[0, :text_embeddings_list[0].shape[0], :] = text_embeddings_list[0]

        # Generate initial noise
        student_config = student_model.config
        latents = torch.randn(
            1, # batch size
            student_config.num_channels,
            student_config.image_size // student_config.patch_size,
            student_config.image_size // student_config.patch_size,
            device=device
        )

        # DDIM sampling loop
        print(f"Running DDIM sampling for {args.num_inference_steps} steps...")
        timesteps = torch.linspace(999, 0, args.num_inference_steps, dtype=torch.long, device=device)
        current_latents = latents
        epsilon = 1e-3

        for i, t in enumerate(timesteps):
            timestep_batch = torch.full((1,), t, device=device, dtype=torch.long)
            
            noise_pred = student_model(current_latents, None, timestep_batch, text_embeddings)
            
            alpha_t = 1.0 - (t.float() / 1000.0)
            alpha_t = torch.clamp(alpha_t, min=epsilon, max=1.0 - epsilon)
            
            alpha_t_prev = 1.0 - (timesteps[i+1].float() / 1000.0) if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            alpha_t_prev = torch.clamp(alpha_t_prev, min=epsilon, max=1.0 - epsilon) if i < len(timesteps) - 1 else alpha_t_prev
            
            pred_original_sample = (current_latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_sample_direction = torch.sqrt(1 - alpha_t_prev) * noise_pred
            current_latents = torch.sqrt(alpha_t_prev) * pred_original_sample + pred_sample_direction

        denoised_latents = current_latents
        print("✓ Denoising complete.")

        # Decode latents
        print("Decoding latents with VAE...")
        vae_latents_list = [denoised_latents[0].unsqueeze(1)] # [C, 1, H, W]
        
        teacher_device = next(teacher_vae.model.parameters()).device
        teacher_dtype = next(teacher_vae.model.parameters()).dtype
        vae_latents_list = [z.to(device=teacher_device, dtype=teacher_dtype) for z in vae_latents_list]

        decoded_videos = teacher_vae.decode(vae_latents_list)
        
        image_tensor = decoded_videos[0][:, 0, :, :]
        image_tensor = (image_tensor.float().cpu().clamp(-1, 1) + 1.0) / 2.0
        
        image = Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))
        
        image.save(args.output_path)
        print(f"✓ Image saved to {args.output_path}")

if __name__ == "__main__":
    main()