import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import WanVideoToVideoPipeline, DDPMScheduler
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import argparse
import json
import os


class StaticPromptsDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.prompts = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


class WanLiteStudent(nn.Module):
    """
    A simplified student architecture based on the provided config.
    Note: This is a scaffold for the architecture. 
    In a real scenario, this would mirror the internal layers of Wan-Lite.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_heads']
        self.depth = config['depth']
        self.patch_size = config['m_patch_size']

        # Example: Embedding layers (Scaffold)
        self.time_embedding = nn.Linear(320, self.hidden_size)
        self.class_embedding = nn.Embedding(1000, self.hidden_size)

        # Transformer Layers (Scaffold)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_size * 4,
                batch_first=True
            )
            for _ in range(self.depth)
        ])

        # Output layer
        self.norm = nn.LayerNorm(self.hidden_size)
        self.proj_out = nn.Linear(self.hidden_size, 320)  # Output channels matching input noise

    def forward(self, noisy_latents, timestep, prompt_embeds):
        # 1. Process Time Embedding
        time_emb = self.time_embedding(timestep)

        # 2. Add embeddings to latents (Simplified concat)
        x = noisy_latents + time_emb.unsqueeze(1)

        # 3. Pass through transformer blocks
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # 4. Predict noise
        noise_pred = self.proj_out(x)

        return noise_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, required=True, help="Path to Wan 2.2 weights")
    parser.add_argument("--student_config", type=str, default="config/student_config.json")
    parser.add_argument("--data_path", type=str, default="data/static_prompts.txt")
    parser.add_argument("--output_dir", type=str, default="./outputs/wan_t2i")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    # Setup Output
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Configuration
    with open(args.student_config, 'r') as f:
        student_config = json.load(f)

    # 2. Load Teacher (Wan 2.2 Video Model)
    # We assume the user has loaded WanVideoToVideoPipeline or similar
    # Ideally, we load the UNet or VAE here as well, but for distillation 
    # we often rely on the pipeline's denoising capabilities.
    print("Loading Teacher (Wan 2.2)...")
    teacher_pipe = WanVideoToVideoPipeline.from_pretrained(args.teacher_path, torch_dtype=torch.float16)
    teacher_pipe.scheduler = DDPMScheduler.from_config(teacher_pipe.scheduler.config)

    # Move teacher to GPU
    teacher_pipe = teacher_pipe.to("cuda")

    # 3. Initialize Student
    print("Initializing Student (Wan-Lite)...")
    student = WanLiteStudent(student_config)
    student = student.to("cuda")

    # 4. Load Data
    dataset = StaticPromptsDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    # 6. Training Loop
    # Logic:
    # 1. Teacher generates high-quality single frame from prompt.
    # 2. We add noise to that image (Diffusion process).
    # 3. Student predicts noise for the noisy image.
    # 4. Compare Student's prediction to the Teacher's "expected" noise (simulated via target generation or just image fidelity).
    # For this scaffold, we use standard distillation: Student predicts noise, Teacher provides the target noise prediction.

    print("Starting Distillation Training...")

    # We'll use the teacher to generate a clean image, then add noise to it
    # and use the teacher's forward pass (simulated) to get the target noise.
    # Note: Actual implementation depends on the exact interface of the Teacher's UNet.

    global_step = 0

    for epoch in range(10):  # Example epoch loop
        for prompts in tqdm(dataloader, desc="Epoch"):
            optimizer.zero_grad()

            # --- Teacher Step (Generating Data) ---
            # Force 1 frame to ignore temporal motion
            with torch.no_grad():
                teacher_output = teacher_pipe(
                    prompt=prompts,
                    num_inference_steps=50,  # High denoise steps for quality
                    num_frames=1,  # Single frame
                    guidance_scale=7.5
                )
                clean_images = teacher_output.images[0]  # Shape: (B, C, H, W)

            # Add Noise to images (Simulating a diffusion timestep)
            # We assume a random timestep t between 0 and 1000 for this batch
            timesteps = torch.randint(0, 1000, (clean_images.size(0),), device="cuda")

            # Add Gaussian Noise
            noise = torch.randn_like(clean_images)
            noisy_images = clean_images + noise * 0.1  # Scaling noise factor

            # Prepare Inputs
            # Note: Input shape usually expects (B, T, C, H, W). Since T=1, squeeze or keep as is.
            noisy_images = noisy_images.unsqueeze(1)
            noise = noise.unsqueeze(1)
            clean_images = clean_images.unsqueeze(1)
            timesteps = timesteps.unsqueeze(1)

            # --- Student Step (Prediction) ---
            # Student predicts noise
            predicted_noise = student(noisy_images.squeeze(1), timesteps.squeeze(1), prompt_embeds=None)

            # In a real scenario, we would compare predicted_noise to the Teacher's predicted_noise
            # For this scaffold, we ensure the shapes match.
            if predicted_noise.shape != noise.shape:
                print(f"Warning: Shape mismatch. Student: {predicted_noise.shape}, Target: {noise.shape}")

            # Loss Function (MSE)
            loss = F.mse_loss(predicted_noise, noise)

            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % 100 == 0:
                print(f"Step {global_step}, Loss: {loss.item():.6f}")

    # Save Student
    save_path = os.path.join(args.output_dir, "distilled_model.pth")
    torch.save(student.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    main()