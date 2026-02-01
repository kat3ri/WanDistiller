import argparse
import json
import os
import sys
import warnings
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from diffusers import DiffusionPipeline
from safetensors.torch import save_file

from projection_mapper import load_and_project_weights


# -----------------------------------------------------------------------------
# Validation and Error Detection
# -----------------------------------------------------------------------------

def check_command_line_usage():
    """
    Check if the script is being run with common mistakes and provide helpful error messages.
    This helps catch issues early before they cause confusing errors later.
    """
    # Check if script name looks like it was run incorrectly with torchrun
    # When someone runs: torchrun --nproc_per_node=4 python train_distillation.py
    # The script name would be 'python' which would cause an error
    script_name = os.path.basename(sys.argv[0])
    
    if script_name == 'python' or script_name == 'python3':
        print("=" * 80)
        print("ERROR: Incorrect usage detected!")
        print("=" * 80)
        print()
        print("It looks like you're trying to run this script with 'torchrun' but")
        print("included 'python' before the script name.")
        print()
        print("When using torchrun, do NOT include 'python' before the script name.")
        print("torchrun already invokes Python internally.")
        print()
        print("✗ Wrong:")
        print("  torchrun --nproc_per_node=4 python train_distillation.py ...")
        print()
        print("✓ Correct:")
        print("  torchrun --nproc_per_node=4 train_distillation.py ...")
        print()
        print("=" * 80)
        sys.exit(1)


# -----------------------------------------------------------------------------
# Multi-GPU / Distributed Training Setup
# -----------------------------------------------------------------------------

def setup_distributed():
    """
    Initialize distributed training process group.
    Returns rank, world_size, and local_rank.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        return 0, 1, 0
    
    # Validate that we have enough GPUs for the requested number of processes
    num_gpus = torch.cuda.device_count()
    if local_rank >= num_gpus:
        print("=" * 80)
        print(f"ERROR: Invalid GPU configuration!")
        print("=" * 80)
        print()
        print(f"This process (rank {rank}, local_rank {local_rank}) is trying to use GPU {local_rank},")
        print(f"but only {num_gpus} GPU(s) are available on this machine (GPU 0 to GPU {num_gpus-1}).")
        print()
        print("This happens when you request more processes than available GPUs.")
        print()
        print(f"✗ Current command uses: --nproc_per_node={os.environ.get('LOCAL_WORLD_SIZE', 'unknown')}")
        print(f"✓ Available GPUs: {num_gpus}")
        print()
        print("Solutions:")
        print(f"  1. Reduce --nproc_per_node to match available GPUs:")
        print(f"     torchrun --nproc_per_node={num_gpus} train_distillation.py ...")
        print()
        print("  2. Or run on CPU without distributed training:")
        print("     python train_distillation.py ... (without --distributed flag)")
        print()
        print("=" * 80)
        sys.exit(1)
    
    # Set the device before initializing the process group
    torch.cuda.set_device(local_rank)
    
    # Initialize process group with explicit device_id to avoid GPU mapping warnings
    # This ensures each process knows which GPU it should use
    # device_id should be an integer representing the local rank
    dist.init_process_group(
        backend='nccl', 
        init_method='env://', 
        world_size=world_size, 
        rank=rank,
        device_id=local_rank
    )
    dist.barrier()
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """
    Clean up distributed training process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """
    Check if current process is the main process (rank 0).
    """
    return rank == 0


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

    def __init__(self, config, teacher_checkpoint_path=None, device=None, distributed=False):
        super().__init__()
        self.config = config
        self.distributed = distributed
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
            if latent_1 is not None and latent_1.dim() == 5:
                latent_1 = latent_1.squeeze(2)

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
    # 0. Check for common command-line usage mistakes
    check_command_line_usage()
    
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
    
    # Multi-GPU arguments
    parser.add_argument("--multi_gpu", action="store_true", 
                        help="Enable multi-GPU training using DataParallel")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training using DistributedDataParallel (recommended for multi-GPU)")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed training (set automatically by torch.distributed.launch)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of data loading workers (default: 0, set to 4+ for better performance)")

    args = parser.parse_args()
    
    # 2. Early validation for distributed training
    if args.distributed:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("=" * 80)
            print("ERROR: Distributed training requires CUDA")
            print("=" * 80)
            print()
            print("You specified --distributed flag, but CUDA is not available.")
            print("Distributed training with NCCL backend requires GPUs.")
            print()
            print("Solutions:")
            print("  1. Run without --distributed flag for CPU training:")
            print("     python train_distillation.py ...")
            print()
            print("  2. Or ensure CUDA is properly installed and GPUs are available")
            print()
            print("=" * 80)
            sys.exit(1)
        
        # Early check for GPU count vs requested processes
        num_gpus = torch.cuda.device_count()
        if 'LOCAL_WORLD_SIZE' in os.environ:
            requested_procs = int(os.environ['LOCAL_WORLD_SIZE'])
            if requested_procs > num_gpus:
                print("=" * 80)
                print("ERROR: Insufficient GPUs for requested processes")
                print("=" * 80)
                print()
                print(f"You are trying to launch {requested_procs} processes (--nproc_per_node={requested_procs})")
                print(f"but only {num_gpus} GPU(s) are available on this machine.")
                print()
                print("Each process needs its own GPU for distributed training.")
                print()
                print("Solutions:")
                print(f"  1. Reduce the number of processes to match available GPUs:")
                print(f"     torchrun --nproc_per_node={num_gpus} train_distillation.py ... --distributed")
                print()
                print("  2. Or run without distributed training:")
                print("     python train_distillation.py ... (without --distributed flag)")
                print()
                if num_gpus > 1:
                    print(f"  3. Or use DataParallel with --multi_gpu (no torchrun needed):")
                    print(f"     python train_distillation.py ... --multi_gpu")
                    print()
                print("=" * 80)
                sys.exit(1)
    
    # 3. Setup distributed training if enabled
    rank = 0
    world_size = 1
    local_rank = args.local_rank
    
    if args.distributed:
        rank, world_size, local_rank = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {rank}] Distributed training initialized: world_size={world_size}, local_rank={local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.multi_gpu and torch.cuda.device_count() > 1:
            print(f"Multi-GPU training enabled with {torch.cuda.device_count()} GPUs using DataParallel")
        print(f"Using device: {device}")
    
    # 4. Load Configuration
    student_config = load_config(args.student_config)

    # Safety check to ensure config loaded successfully
    if student_config is None:
        if is_main_process(rank):
            print("Error: Could not load student configuration.")
        if args.distributed:
            cleanup_distributed()
        return

    # 5. Initialize Student Model
    # Initialize with teacher checkpoint path to enable weight projection
    student_model = WanLiteStudent(
        student_config, 
        teacher_checkpoint_path=args.teacher_path, 
        device=device,
        distributed=args.distributed
    )
    
    # Apply multi-GPU wrapper if enabled
    if args.distributed:
        # Use DistributedDataParallel for better performance
        student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process(rank):
            print(f"Model wrapped with DistributedDataParallel on {world_size} GPUs")
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        # Use DataParallel for simple multi-GPU
        student_model = nn.DataParallel(student_model)
        if is_main_process(rank):
            print(f"Model wrapped with DataParallel on {torch.cuda.device_count()} GPUs")

    # 6. Initialize Teacher Model (Diffusers)
    # This loads the model from the path specified
    # Only print from main process or if not using distributed training
    if is_main_process(rank) or not args.distributed:
        print(f"Loading teacher model from: {args.teacher_path}")

    teacher_pipe = None
    # In environments without network access or when the model is not cached,
    # we'll use a mock teacher model instead
    try:
        # First try local_files_only to avoid network timeout
        if is_main_process(rank):
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

            if is_main_process(rank):
                print(f"   Loading pipeline components (this may take a few minutes)...")
            teacher_pipe = DiffusionPipeline.from_pretrained(
                args.teacher_path,
                **load_kwargs
            )

        if is_main_process(rank):
            print(f"   Moving pipeline to {device}...")
        teacher_pipe.to(device)
        if is_main_process(rank):
            print("✓ Loaded real teacher model from local cache")

        # Note about UMT5 text encoder
        if hasattr(teacher_pipe, 'text_encoder') and is_main_process(rank):
            print("   Note: T5/UMT5 models use tied weights (shared.weight ↔ encoder.embed_tokens.weight)")
            print("         Any 'MISSING' warnings for embed_tokens are expected and can be ignored.")

    except Exception as e:
        if is_main_process(rank):
            print(f"   Could not load from local cache: {str(e)[:100]}")
        try:
            # Try with network access (if available)
            if is_main_process(rank):
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

                if is_main_process(rank):
                    print(f"   Loading pipeline components (this may take a few minutes)...")
                teacher_pipe = DiffusionPipeline.from_pretrained(
                    args.teacher_path,
                    **load_kwargs
                )

            if is_main_process(rank):
                print(f"   Moving pipeline to {device}...")
            teacher_pipe.to(device)
            if is_main_process(rank):
                print("✓ Loaded real teacher model from HuggingFace")

            # Note about UMT5 text encoder
            if hasattr(teacher_pipe, 'text_encoder') and is_main_process(rank):
                print("   Note: T5/UMT5 models use tied weights (shared.weight ↔ encoder.embed_tokens.weight)")
                print("         Any 'MISSING' warnings for embed_tokens are expected and can be ignored.")

        except Exception as e2:
            if is_main_process(rank):
                print(f"   Could not download: {str(e2)[:100]}")
            teacher_pipe = None

    # Exit with error if real model couldn't be loaded
    if teacher_pipe is None:
        if is_main_process(rank):
            print("\n" + "=" * 80)
            print("ERROR: Failed to load teacher model")
            print("=" * 80)
            print(f"Model path: {args.teacher_path}")
            print("\nThe teacher model must be available to run distillation training.")
            print("Please ensure:")
            print("  1. The model is downloaded and cached locally, OR")
            print("  2. You have internet access to download from HuggingFace Hub")
            print("\nTo cache the model locally, you can download it using:")
            print("  from diffusers import DiffusionPipeline")
            print(f"  DiffusionPipeline.from_pretrained('{args.teacher_path}')")
            print("=" * 80)
        if args.distributed:
            cleanup_distributed()
        sys.exit(1)

    # Wan2.1 models usually have the transformer as a specific attribute
    # Adjust this if your Diffusers model structure is different
    teacher_model = teacher_pipe.transformer
    teacher_model.eval()  # Teacher should not be trained
    # Add this inside your main loop or initialization
    if is_main_process(rank):
        print("--- Teacher Model Config ---")
        print(f"Expected in_channels (config): {teacher_model.config.get('in_channels', 'Not found')}")

        # Look for projection layers
        print(f"Has proj_in: {hasattr(teacher_model, 'proj_in')}")
        if hasattr(teacher_model, 'proj_in'):
            print(f"Proj in shape: {teacher_model.proj_in.weight.shape}")

        print(f"Has patch_embedding: {hasattr(teacher_model, 'patch_embedding')}")
        if hasattr(teacher_model, 'patch_embedding'):
            print(f"Patch embedding shape: {teacher_model.patch_embedding.weight.shape}")

    # 7. Initialize Dataset and Dataloader
    dataset = StaticPromptsDataset(args.data_path)
    
    # Use DistributedSampler for distributed training
    if args.distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            sampler=sampler,
            num_workers=args.num_workers,  # Configurable number of workers
            pin_memory=True
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers  # Configurable number of workers
        )

    # 8. Optimizer and Loss
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
    loss_fn = torch.nn.functional.mse_loss

    # 9. Training Loop
    student_model.train()  # Student needs to be in train mode
    global_step = 0

    if is_main_process(rank):
        print("Starting distillation training...")

    # Fixed: Use args.num_epochs
    for epoch in range(args.num_epochs):
        # Set epoch for DistributedSampler to ensure different shuffling each epoch
        if args.distributed:
            sampler.set_epoch(epoch)
            
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
                    # num_images_per_prompt=1,
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

            # Save original latents for student (before any projection)
            # Student expects original num_channels (4), not projected channels
            student_latents = latents.clone()

            with torch.no_grad():
                # 1. Ensure Latent Shape (B, C, T, H, W)
                # This assumes your latent shape is currently (B, 4, T, H, W)
                if latents.dim() == 4:
                    latents = latents.unsqueeze(2)

                current_channels = latents.shape[1]

                # 2. Extract Expected Channels from Config
                # Based on your printout: Expected in_channels (config): 16
                expected_channels = teacher_model.config.get('in_channels', 4)

                # This handles the mismatch between the model's expectation and the device dtype
                latents = latents.float()
                text_embeddings = text_embeddings.float()

                # 3. Inject Projection Layer if Mismatch Exists
                if current_channels != expected_channels:
                    print(f"Projection needed: {current_channels} -> {expected_channels}")

                    # Define a Conv3d projection layer
                    proj_layer = nn.Conv3d(
                        in_channels=current_channels,
                        out_channels=expected_channels,
                        kernel_size=1
                    )

                    # Initialize weights to 0
                    nn.init.zeros_(proj_layer.weight)
                    if proj_layer.bias is not None:
                        nn.init.zeros_(proj_layer.bias)

                    # --- FIX ---
                    # Cast the projection layer to Float32 explicitly
                    proj_layer = proj_layer.float()

                    # Move to device (latents.device is now effectively Float32 due to .float() above)
                    proj_layer = proj_layer.to(latents.device)

                    # Apply projection (only for teacher, not student)
                    latents = proj_layer(latents)

                # 4. Pass through teacher
                # Cast timesteps to float32 to satisfy Linear layer requirements
                teacher_output = teacher_model(
                    hidden_states=latents,
                    timestep=timesteps.float(),  # <--- Ensure timestep is also float32
                    encoder_hidden_states=text_embeddings,
                )

                # ... (Extract output logic) ...

                # Extract the actual tensor from teacher output
                # Diffusers models often return a dict or object with .sample attribute
                if isinstance(teacher_output, dict):
                    teacher_output = teacher_output.get('sample', teacher_output)
                elif hasattr(teacher_output, 'sample'):
                    teacher_output = teacher_output.sample

            # --- STUDENT PASS ---

            # Use original latents for student (not the projected ones used for teacher)
            # Student conv_in expects config['num_channels'], not teacher's in_channels
            student_output = student_model(
                latent_0=student_latents,
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

            if global_step % 50 == 0 and is_main_process(rank):
                print(f"Step {global_step}: Loss = {loss.item()}")

    # 10. Save Model (only from main process)
    if is_main_process(rank):
        # Unwrap model if using DataParallel or DistributedDataParallel
        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
        model_to_save.save_pretrained(args.output_dir)
        print("Training complete.")
    
    # Clean up distributed training
    if args.distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
