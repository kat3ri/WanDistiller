import argparse
import json
import os
import sys
import warnings
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from diffusers import DiffusionPipeline
from safetensors.torch import save_file

from projection_mapper import load_and_project_weights



# -----------------------------------------------------------------------------
# Dataset Configuration
# -----------------------------------------------------------------------------

class StaticPromptsDataset(torch.utils.data.Dataset):
    """
    Dataset to load prompts from a text file.
    """

    def __init__(self, file_path, tokenizer=None, max_length=77):
        self.prompts = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load prompts from the provided text file
        with open(file_path, 'r', encoding='utf-8') as f:
            self.prompts = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


# -----------------------------------------------------------------------------
# Model Definition
# -----------------------------------------------------------------------------

class WanTransformerBlock(nn.Module):
    """
    A basic Transformer block component.
    Injects time and text embeddings into the processing.
    """

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)

        # MLP usually takes time embedding to scale its activation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

    def forward(self, x, t_emb, text_emb):
        """
        x: image latent (batch, seq, hidden_size)
        t_emb: timestep embedding (batch, hidden_size)
        text_emb: text embedding (batch, hidden_size)
        """

        # 1. Add text embedding to the input (Conditioning)
        # We add text_emb to x before processing
        x = x + text_emb.unsqueeze(1)

        # 2. Add time embedding to the input (Conditioning)
        # This is the missing step. We add t_emb to x to condition the block
        # on the current diffusion timestep.
        x = x + t_emb.unsqueeze(1)

        # Attention Block
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output

        # MLP Block
        mlp_input = self.norm2(x)

        # You can also use t_emb here if you want to scale the MLP activation,
        # but simple addition is the standard DiT approach.

        mlp_output = self.mlp(mlp_input)
        x = x + mlp_output

        return x


class WanLiteStudent(nn.Module):
    """
    WanLiteStudent model definition - A 2D image-only model.
    
    This model is designed for static image generation (Text-to-Image),
    not video generation. It receives projected weights from the teacher
    video model (Wan 2.2) but strips out all temporal/motion components.
    
    The projection_mapper handles converting the teacher's 3D video weights
    to the student's 2D image architecture.
    """

    def __init__(self, config, teacher_checkpoint_path=None, device=None):
        super().__init__()
        self.config = config
        # Default to cuda if available, otherwise cpu
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Extract configuration parameters
        hidden_size = config["hidden_size"]
        depth = config["depth"]
        num_heads = config["num_heads"]
        num_channels = config["num_channels"]
        text_encoder_output_dim = config["text_encoder_output_dim"]

        # 1. Text Projection Layer
        # Maps text encoder output (e.g., 4096) down to hidden size (e.g., 640)
        # Defined here to be populated by load_and_project_weights
        self.text_proj = nn.Linear(text_encoder_output_dim, hidden_size)

        # 2. Time Embedding (for diffusion timesteps)
        time_embed_dim = hidden_size * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_size)
        )

        # 3. Convolutional Layers (2D only - no temporal dimension)
        self.conv_in = nn.Conv2d(num_channels, hidden_size, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(hidden_size, num_channels, kernel_size=3, padding=1)

        # 4. Transformer Blocks (2D spatial attention only)
        self.blocks = nn.ModuleList([
            WanTransformerBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # Move to device first
        self.to(device)
        
        # 5. Apply Projection Mapper
        # Loads weights from the teacher checkpoint and maps them to the student architecture
        # This handles converting 3D video model weights to 2D image model weights
        if teacher_checkpoint_path is not None:
            print(f"Loading teacher weights from: {teacher_checkpoint_path}")
            # For now, skip the projection mapper as it expects state_dict not a path
            # The projection mapper would need to be updated to handle HuggingFace model loading
            print("Note: Weight projection from teacher model is not yet implemented for HuggingFace models")

    def forward(self, latent_0, latent_1, timestep, encoder_hidden_states):
        """
        Forward pass for 2D image generation (no temporal/video dimensions).
        
        Args:
            latent_0: Input latent tensor (batch, channels, height, width) - 2D spatial only
            latent_1: Optional conditioning latent (batch, channels, height, width) or None
            timestep: Diffusion timestep (batch,)
            encoder_hidden_states: Text embeddings (batch, seq_len, text_dim)
        
        Returns:
            Output prediction (batch, channels, height, width) - 2D spatial only
        """

        # 1. Convolutional Input (2D spatial only)
        # We add latent_1 to the input if it is a conditioning latent
        # (depending on architecture, latent_1 might be added here or passed into blocks)
        x = self.conv_in(latent_0)
        if latent_1 is not None:
            x = x + latent_1

        # 2. Time Embedding
        # Convert timestep to float and create sinusoidal embeddings
        if timestep.dtype != torch.float32:
            timestep = timestep.float()
        
        # Create sinusoidal time embeddings
        batch_size = x.shape[0]
        hidden_size = self.config["hidden_size"]
        
        # Ensure hidden_size is even for proper sin/cos concatenation
        if hidden_size % 2 != 0:
            raise ValueError(f"hidden_size must be even for time embeddings, got {hidden_size}")
        
        half_dim = hidden_size // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        t_emb = self.time_embed(emb)

        # 3. Text Embedding Projection
        # Average pool the text embeddings to get a single vector per batch
        text_emb = self.text_proj(encoder_hidden_states.mean(dim=1))

        # 4. Transformer Processing (2D spatial attention only)
        # Reshape x for transformer: (batch, channels, h, w) -> (batch, h*w, channels)
        batch, channels, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (batch, h*w, channels)
        
        # We must pass t_emb and text_emb to every block now
        for block in self.blocks:
            x = block(x, t_emb, text_emb)
        
        # Reshape back: (batch, h*w, channels) -> (batch, channels, h, w)
        x = x.transpose(1, 2).reshape(batch, channels, h, w)

        # 5. Convolutional Output (2D spatial only)
        student_output = self.conv_out(x)
        return student_output

    def save_pretrained(self, output_dir):
        """
        Saves the model as a Diffusers-compatible structure.
        Saves:
        1. student_config.json (The configuration)
        2. diffusion_model.safetensors (The weights)
        """

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save Configuration as JSON
        # Diffusers requires the config to be a JSON file
        config = self.config
        config_path = os.path.join(output_dir, "student_config.json")

        # Ensure specific keys match Diffusers expectations if necessary,
        # or just save the current config.
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        # 2. Save Weights as Safetensors
        # This is the standard format for Diffusers
        weights_path = os.path.join(output_dir, "diffusion_model.safetensors")

        # Note: We only save the state_dict here.
        # You might need to filter keys here if the state_dict has extra keys
        # (like 'epoch' or 'optimizer') that you don't want to save.
        save_file(self.state_dict(), weights_path)

        print(f"Model saved successfully to: {output_dir}")




# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

import json


def load_config(json_path):
    """
    Load the student configuration from the provided JSON file.
    Returns a dictionary containing model architecture and shape parameters.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        config_vars = {
            "model_type": data.get("model_type"),
            "hidden_size": data.get("hidden_size"),
            "depth": data.get("depth"),
            "num_heads": data.get("num_heads"),
            "num_channels": data.get("num_channels"),
            "image_size": data.get("image_size"),
            "patch_size": data.get("patch_size"),
            "text_max_length": data.get("text_max_length"),
            "text_encoder_output_dim": data.get("text_encoder_output_dim"),
            "projection_factor": data.get("projection_factor")
        }

        return config_vars

    except FileNotFoundError:
        print(f"Error: Config file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Config file contains invalid JSON syntax.")
        return None

# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------

def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--student_config", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=10)  # Added this argument

    args = parser.parse_args()

    # 2. Load Configuration
    student_config = load_config(args.student_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Safety check to ensure config loaded successfully
    if student_config is None:
        print("Error: Could not load student configuration.")
        return

    # 3. Initialize Student Model
    # Initialize with teacher checkpoint path to enable weight projection
    student_model = WanLiteStudent(student_config, teacher_checkpoint_path=args.teacher_path, device=device)

    # 4. Initialize Teacher Model (Diffusers)
    # This loads the model from the path specified
    print(f"Loading teacher model from: {args.teacher_path}")
    
    teacher_pipe = None
    # In environments without network access or when the model is not cached,
    # we'll use a mock teacher model instead
    try:
        # First try local_files_only to avoid network timeout
        print("   Trying local cache first...")
        
        # Suppress warnings about tied weights in T5/UMT5 models
        # These warnings are expected and harmless - encoder.embed_tokens.weight
        # is tied to shared.weight, so the "MISSING" warning is misleading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*were not used when initializing.*")
            warnings.filterwarnings("ignore", message=".*were newly initialized.*")
            
            # Load the model in its native format without forcing dtype or variant
            # This allows the model to load in whatever format is available
            # Note: Model will load in full precision (fp32) which requires more GPU memory
            # but ensures compatibility when fp16 variant is not available
            load_kwargs = {
                "local_files_only": True,
            }
            
            print(f"   Loading pipeline components (this may take a few minutes)...")
            teacher_pipe = DiffusionPipeline.from_pretrained(
                args.teacher_path,
                **load_kwargs
            )
        
        print(f"   Moving pipeline to {device}...")
        teacher_pipe.to(device)
        print("✓ Loaded real teacher model from local cache")
        
        # Note about UMT5 text encoder
        if hasattr(teacher_pipe, 'text_encoder'):
            print("   Note: T5/UMT5 models use tied weights (shared.weight ↔ encoder.embed_tokens.weight)")
            print("         Any 'MISSING' warnings for embed_tokens are expected and can be ignored.")
            
    except Exception as e:
        print(f"   Could not load from local cache: {str(e)[:100]}")
        try:
            # Try with network access (if available)
            print("   Trying to download from HuggingFace Hub...")
            
            # Suppress warnings about tied weights
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*were not used when initializing.*")
                warnings.filterwarnings("ignore", message=".*were newly initialized.*")
                
                # Load the model in its native format without forcing dtype or variant
                # This allows the model to load in whatever format is available
                # Note: Model will load in full precision (fp32) which requires more GPU memory
                # but ensures compatibility when fp16 variant is not available
                load_kwargs = {
                    "local_files_only": False,
                }
                
                print(f"   Loading pipeline components (this may take a few minutes)...")
                teacher_pipe = DiffusionPipeline.from_pretrained(
                    args.teacher_path,
                    **load_kwargs
                )
            
            print(f"   Moving pipeline to {device}...")
            teacher_pipe.to(device)
            print("✓ Loaded real teacher model from HuggingFace")
            
            # Note about UMT5 text encoder
            if hasattr(teacher_pipe, 'text_encoder'):
                print("   Note: T5/UMT5 models use tied weights (shared.weight ↔ encoder.embed_tokens.weight)")
                print("         Any 'MISSING' warnings for embed_tokens are expected and can be ignored.")
                
        except Exception as e2:
            print(f"   Could not download: {str(e2)[:100]}")
            teacher_pipe = None
    
    # Exit with error if real model couldn't be loaded
    if teacher_pipe is None:
        print("\n" + "="*80)
        print("ERROR: Failed to load teacher model")
        print("="*80)
        print(f"Model path: {args.teacher_path}")
        print("\nThe teacher model must be available to run distillation training.")
        print("Please ensure:")
        print("  1. The model is downloaded and cached locally, OR")
        print("  2. You have internet access to download from HuggingFace Hub")
        print("\nTo cache the model, run:")
        print(f"  python -c \"from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('{args.teacher_path}')\"")
        print("="*80)
        sys.exit(1)
    

    # Wan2.1 models usually have the transformer as a specific attribute
    # Adjust this if your Diffusers model structure is different
    teacher_model = teacher_pipe.transformer
    teacher_model.eval()  # Teacher should not be trained

    # 5. Initialize Dataset and Dataloader
    dataset = StaticPromptsDataset(args.data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 6. Optimizer and Loss
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
    loss_fn = torch.nn.functional.mse_loss

    # 7. Training Loop
    student_model.train()  # Student needs to be in train mode
    global_step = 0

    print("Starting distillation training...")

    # Fixed: Use args.num_epochs
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            # --- DATA PREPARATION ---
            # batch is a list of prompt strings
            prompts = batch
            
            # Encode the prompts using the teacher's text encoder
            # For Diffusers models, text encoding is usually done via encode_prompt or similar
            # We'll use a simple approach assuming teacher_pipe has text_encoder
            if hasattr(teacher_pipe, 'encode_prompt'):
                # Use encode_prompt if available
                text_embeddings = teacher_pipe.encode_prompt(
                    prompts, 
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False
                )[0]
            elif hasattr(teacher_pipe, 'text_encoder'):
                # Fallback to direct text encoder usage
                if hasattr(teacher_pipe, 'tokenizer'):
                    text_inputs = teacher_pipe.tokenizer(
                        prompts,
                        padding="max_length",
                        max_length=teacher_pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    text_embeddings = teacher_pipe.text_encoder(text_inputs.input_ids)[0]
                else:
                    raise ValueError("Text encoder found but no tokenizer available")
            else:
                raise ValueError("No text encoder found in teacher pipeline")

            # Generate random timesteps and latents for the step
            # Shape: [batch_size, 16, 16, 4] (Example based on patch_size=16 and num_channels=4)
            latents = torch.randn(
                args.batch_size,
                student_config["num_channels"],
                student_config["image_size"] // student_config["patch_size"],
                student_config["image_size"] // student_config["patch_size"],
                device=device
            )

            # Generate random timesteps
            timesteps = torch.randint(0, 1000, (args.batch_size,), device=device).long()

            # --- TEACHER PASS ---
            with torch.no_grad():
                # Pass through teacher
                teacher_output = teacher_model(
                    sample=latents,
                    timestep=timesteps,
                    encoder_hidden_states=text_embeddings,
                )
                
                # Extract the actual tensor from teacher output
                # Diffusers models often return a dict or object with .sample attribute
                if isinstance(teacher_output, dict):
                    teacher_output = teacher_output.get('sample', teacher_output)
                elif hasattr(teacher_output, 'sample'):
                    teacher_output = teacher_output.sample

            # --- STUDENT PASS ---
            student_output = student_model(
                latent_0=latents,
                latent_1=None,
                timestep=timesteps,
                encoder_hidden_states=text_embeddings,
            )

            # --- LOSS CALCULATION ---
            loss = loss_fn(student_output, teacher_output)

            # --- BACKPROPAGATION ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 50 == 0:
                print(f"Step {global_step}: Loss = {loss.item()}")

    # 8. Save Model
    student_model.save_pretrained(args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
