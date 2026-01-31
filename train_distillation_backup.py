import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import WanVideoToVideoPipeline, DDPMScheduler, WanVaeImageToVideoPipeline
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer
from tqdm.auto import tqdm
import argparse
import json
import os
import math


# --- 1. Configuration Helper ---
def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


# --- 2. Data Loader ---
class StaticPromptsDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=77):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.prompts = [line.strip() for line in f if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        text = self.prompts[idx]
        # Tokenize and truncate/pad to max_length
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        return inputs["input_ids"].squeeze(0)


# --- 3. Student Model Architecture (DiT / UNet style) ---
class WanLiteStudent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Extract dimensions from config
        d_model = config['hidden_size']
        depth = config['depth']
        m_channels = config['m_channels']
        m_patch_size = config['m_patch_size']

        # --- Embeddings ---
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )

        # CLIP projection layer (Simplified)
        self.text_proj = nn.Linear(d_model, d_model)

        # --- Downsample / Upsample Layers (Simplified for Scaffold) ---
        # Input: B x C x H x W
        self.conv_in = nn.Conv2d(4, d_model, kernel_size=3, padding=1)

        # Blocks (Simplified Transformer/Conv combination)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(d_model, num_heads=config['num_heads'], batch_first=True),
                "ff": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.SiLU(),
                    nn.Linear(d_model * 4, d_model)
                ),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
            })
            for _ in range(depth)
        ])

        # Output Layer
        self.conv_out = nn.Conv2d(d_model, 4, kernel_size=3, padding=1)

    def forward(self, sample, timestep, encoder_hidden_states):
        """
        sample: noisy latents (B, 4, H, W)
        timestep: diffusion timestep (B)
        encoder_hidden_states: text embeddings (B, L, D)
        """
        # 1. Process Timestep
        # Ensure timestep is float for embedding
        c = timestep.type(sample.dtype)
        temb = self.time_embed(c)

        # Add time embedding to sample (Spatial attention)
        sample = sample + temb.unsqueeze(-1).unsqueeze(-1)

        # 2. Process Text
        encoder_hidden_states = self.text_proj(encoder_hidden_states)

        # 3. Pass through blocks
        for block in self.blocks:
            # Attention
            attn_out = block["attn"](encoder_hidden_states, encoder_hidden_states, encoder_hidden_states)[0]
            encoder_hidden_states = block["norm1"](encoder_hidden_states) + attn_out

            # Feed Forward
            ffw_out = block["ff"](encoder_hidden_states)
            encoder_hidden_states = block["norm2"](encoder_hidden_states) + ffw_out

        # 4. Output projection (skip connection logic omitted for brevity in scaffold)
        # In a real model, we project back to spatial features or use UNet structure
        # Here we assume a simplified mapping or simply pass spatial latents through
        return self.conv_out(encoder_hidden_states)


# --- 4. Training Main Logic ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--student_config", type=str, default="config/student_config.json")
    parser.add_argument("--data_path", type=str, default="data/static_prompts.txt")
    parser.add_argument("--output_dir", type=str, default="./outputs/wan_t2i")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Configuration
    student_config = load_config(args.student_config)

    # 2. Load Teacher Components
    print("Loading Teacher Components...")
    teacher_pipe = WanVideoToVideoPipeline.from_pretrained(args.teacher_path, torch_dtype=torch.float16, variant="fp16")

    # We only need the VAE and the specific UNet for training logic usually,
    # but keeping the pipeline allows easy generation if needed later.
    teacher_pipe = teacher_pipe.to("cuda")

    # Load VAE for encoding/decoding
    vae = teacher_pipe.vae
    vae = vae.to("cuda")
    vae_dtype = torch.float16
    vae = vae.to(vae_dtype)

    # Load Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # 3. Initialize Student
    print("Initializing Student...")
    student = WanLiteStudent(student_config)
    student = student.to("cuda")

    # 4. Load Data
    dataset = StaticPromptsDataset(args.data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    noise_scheduler = DDPMScheduler.from_config(teacher_pipe.scheduler.config)

    # 6. Training Loop
    print("Starting Distillation Training...")

    global_step = 0

    for epoch in range(10):
        for prompt_ids in tqdm(dataloader, desc="Epoch"):
            optimizer.zero_grad()
            device = "cuda"

            # --- A. Text Embedding (CLIP) ---
            with torch.no_grad():
                text_inputs = tokenizer(
                    [prompt.tolist() for prompt in prompt_ids],
                    # Decode list back to string for tokenizer if necessary, or just use raw ids if CLIP handles it
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77
                )
                text_input_ids = text_inputs.input_ids.to(device)
                text_embeddings = teacher_pipe.text_encoder(text_input_ids.to(vae_dtype))[0]

            # We use text_embeddings for the student as well
            encoder_hidden_states = text_embeddings

            # --- B. Teacher Generation (Data Prep) ---
            # Teacher generates a clean image from the prompt
            with torch.no_grad():
                teacher_output = teacher_pipe(
                    prompt=prompt_ids.tolist(),  # Passing prompt ids if pipeline accepts them, or text
                    num_inference_steps=50,
                    num_frames=1,
                    guidance_scale=1.0  # Low guidance for dataset gen
                )
                clean_image = teacher_output.images[0].to(device, vae_dtype)

            # Encode clean image to latent space (VAE)
            latents = vae.encode(clean_image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # --- C. Diffusion Forward Pass ---
            # Sample a random noise level
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                      device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents
            noisy_latents = noise_scheduler.add_noise(latents, torch.randn_like(latents), timesteps)

            # --- D. Student Prediction ---
            # Student predicts noise
            noise_pred_student = student(noisy_latents, timesteps, encoder_hidden_states)

            # We don't have a teacher UNet here explicitly, so we simulate the target noise
            # In a real scenario: noise_pred_teacher = teacher_unet(noisy_latents, timesteps, encoder_hidden_states)
            # Here we ensure Student matches the random noise we added, scaled by the scheduler's variance?
            # Actually, standard distillation is MSE(Noise_Pred_Student, Noise_Pred_Teacher).
            # Since we simulated the diffusion step, the "target" noise is the one the student *should* have learned to cancel out.
            # But for this scaffold, we assume the Teacher's VAE pipeline provides the distribution.
            # Let's assume we want the student to reconstruct the latents.

            # For a pure noise prediction distillation, we compare student noise prediction to the actual noise added,
            # adjusted for the noise scheduler variance if necessary.
            # Simplified: The Student tries to predict the Gaussian noise added.

            target_noise = torch.randn_like(latents)

            # Note: DDPMScheduler variance formula is complex.
            # Ideally, we use the scheduler's get_view_as_noise or similar to get the target noise prediction.
            # For this scaffold, we stick to the student trying to predict the added noise.

            loss = F.mse_loss(noise_pred_student, target_noise)

            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % 100 == 0:
                print(f"Step {global_step}, Loss: {loss.item():.6f}")

    # Save Student
    save_path = os.path.join(args.output_dir, "distilled_student.pth")
    torch.save(student.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    main()