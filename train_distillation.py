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
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file
from easydict import EasyDict

from projection_mapper import load_and_project_weights

# Import WAN modules for teacher model
from wan.text2video import WanT2V
from wan.configs.wan_t2v_A14B import t2v_A14B


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
        print("=" * 80, file=sys.stderr)
        print("ERROR: Incorrect usage detected!", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(file=sys.stderr)
        print("It looks like you're trying to run this script with 'torchrun' but", file=sys.stderr)
        print("included 'python' before the script name.", file=sys.stderr)
        print(file=sys.stderr)
        print("When using torchrun, do NOT include 'python' before the script name.", file=sys.stderr)
        print("torchrun already invokes Python internally.", file=sys.stderr)
        print(file=sys.stderr)
        print("✗ Wrong:", file=sys.stderr)
        print("  torchrun --nproc_per_node=4 python train_distillation.py ...", file=sys.stderr)
        print(file=sys.stderr)
        print("✓ Correct:", file=sys.stderr)
        print("  torchrun --nproc_per_node=4 train_distillation.py ...", file=sys.stderr)
        print(file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        sys.exit(1)


# -----------------------------------------------------------------------------
# Multi-GPU / Distributed Training Setup
# -----------------------------------------------------------------------------

def is_main_process(rank):
    """
    Check if current process is the main process (rank 0).
    """
    return rank == 0


def cleanup_distributed():
    """
    Clean up distributed training process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()

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
    
    # Check if CUDA is available before proceeding
    if not torch.cuda.is_available():
        print("=" * 80, file=sys.stderr)
        print(f"[Rank {rank}] ERROR: CUDA is not available!", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(file=sys.stderr)
        print("Distributed training with NCCL backend requires CUDA-enabled GPUs.", file=sys.stderr)
        print(file=sys.stderr)
        print("Solutions:", file=sys.stderr)
        print("  1. Ensure CUDA is properly installed:", file=sys.stderr)
        print("     - Check: nvidia-smi", file=sys.stderr)
        print("     - Reinstall PyTorch with CUDA support if needed", file=sys.stderr)
        print(file=sys.stderr)
        print("  2. Run without distributed training:", file=sys.stderr)
        print("     python train_distillation.py ... (remove torchrun)", file=sys.stderr)
        print(file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        sys.exit(1)
    
    # Validate that we have enough GPUs for the requested number of processes
    num_gpus = torch.cuda.device_count()
    if local_rank >= num_gpus:
        print("=" * 80, file=sys.stderr)
        print(f"[Rank {rank}] ERROR: Invalid GPU configuration!", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(file=sys.stderr)
        print(f"This process (rank {rank}, local_rank {local_rank}) is trying to use GPU index {local_rank},", file=sys.stderr)
        print(f"but only {num_gpus} GPU(s) are available on this machine (GPU indices 0 to {num_gpus-1}).", file=sys.stderr)
        print(file=sys.stderr)
        print("This happens when you request more processes than available GPUs.", file=sys.stderr)
        print(file=sys.stderr)
        print(f"✗ Current command uses: --nproc_per_node={os.environ.get('LOCAL_WORLD_SIZE', 'unknown')}", file=sys.stderr)
        print(f"✓ Available GPUs: {num_gpus}", file=sys.stderr)
        print(file=sys.stderr)
        print("Solutions:", file=sys.stderr)
        print(f"  1. Reduce --nproc_per_node to match available GPUs:", file=sys.stderr)
        print(f"     torchrun --nproc_per_node={num_gpus} train_distillation.py ...", file=sys.stderr)
        print(file=sys.stderr)
        print("  2. Or run on CPU without distributed training:", file=sys.stderr)
        print("     python train_distillation.py ... (without --distributed flag)", file=sys.stderr)
        print(file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        sys.exit(1)
    
    # Set the device before initializing the process group
    # This must be done before init_process_group to ensure proper GPU mapping
    try:
        torch.cuda.set_device(local_rank)
    except Exception as e:
        print("=" * 80, file=sys.stderr)
        print(f"[Rank {rank}] ERROR: Failed to set CUDA device {local_rank}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        print(file=sys.stderr)
        print("This usually means:", file=sys.stderr)
        print(f"  - GPU {local_rank} is not accessible", file=sys.stderr)
        print(f"  - CUDA driver issue", file=sys.stderr)
        print(f"  - GPU {local_rank} is being used by another process", file=sys.stderr)
        print(file=sys.stderr)
        print("Solutions:", file=sys.stderr)
        print("  1. Check GPU status: nvidia-smi", file=sys.stderr)
        print("  2. Free up GPU memory or use a different GPU", file=sys.stderr)
        print("  3. Restart CUDA driver if needed", file=sys.stderr)
        print(file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        sys.exit(1)
    
    # Initialize the distributed process group
    # The NCCL backend will use the device set by torch.cuda.set_device()
    try:
        dist.init_process_group(
            backend='nccl', 
            init_method='env://', 
            world_size=world_size, 
            rank=rank
        )
        dist.barrier()
    except Exception as e:
        print("=" * 80, file=sys.stderr)
        print(f"[Rank {rank}] ERROR: Failed to initialize distributed process group", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        print(file=sys.stderr)
        print("Common causes:", file=sys.stderr)
        print("  1. NCCL backend not properly installed or configured", file=sys.stderr)
        print("  2. Network issues preventing inter-process communication", file=sys.stderr)
        print("  3. Mismatched PyTorch/CUDA versions", file=sys.stderr)
        print("  4. Environment variables not properly set by torchrun", file=sys.stderr)
        print(file=sys.stderr)
        print("Current environment:", file=sys.stderr)
        print(f"  RANK={rank}", file=sys.stderr)
        print(f"  WORLD_SIZE={world_size}", file=sys.stderr)
        print(f"  LOCAL_RANK={local_rank}", file=sys.stderr)
        print(f"  MASTER_ADDR={os.environ.get('MASTER_ADDR', 'not set')}", file=sys.stderr)
        print(f"  MASTER_PORT={os.environ.get('MASTER_PORT', 'not set')}", file=sys.stderr)
        print(file=sys.stderr)
        print("Solutions:", file=sys.stderr)
        print("  1. Verify PyTorch is built with NCCL support:", file=sys.stderr)
        print("     python -c 'import torch; print(torch.cuda.nccl.is_available())'", file=sys.stderr)
        print(file=sys.stderr)
        print("  2. Ensure proper network configuration if using multiple nodes", file=sys.stderr)
        print(file=sys.stderr)
        print("  3. Check firewall settings if communication fails", file=sys.stderr)
        print(file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        sys.exit(1)
    
    return rank, world_size, local_rank


def print_gpu_memory_summary(rank=0, prefix=""):
    """
    Print GPU memory usage summary for debugging OOM issues.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # Convert to GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        
        print(f"[Rank {rank}] {prefix} GPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Max Allocated: {max_allocated:.2f} GB")
        print(f"  Total: {total:.2f} GB")
        print(f"  Free: {total - allocated:.2f} GB")


# -----------------------------------------------------------------------------
# Dataset Configuration
# -----------------------------------------------------------------------------

class StaticPromptsDataset(Dataset):
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

    def __init__(self, config, teacher_checkpoint_path=None, device=None, distributed=False, use_gradient_checkpointing=False):
        super().__init__()
        self.config = config
        self.distributed = distributed
        self.use_gradient_checkpointing = use_gradient_checkpointing
        # Default to cuda if available, otherwise cpu
        if device is None:
            if distributed:
                # In a distributed setting, device should NOT be None.
                # It should be explicitly passed as cuda:local_rank.
                raise ValueError("Device must be specified for distributed training.")
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
        # Cast input to the layer's weight dtype to handle mixed precision from the teacher.
        text_input_for_proj = encoder_hidden_states.mean(dim=1).to(dtype=self.text_proj.weight.dtype)
        text_emb = self.text_proj(text_input_for_proj)

        # 4. Transformer Processing (2D spatial attention only)
        # Reshape x for transformer: (batch, channels, h, w) -> (batch, h*w, channels)
        batch, channels, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (batch, h*w, channels)

        # MEMORY OPTIMIZATION: Use gradient checkpointing if enabled
        # This trades compute for memory by not storing intermediate activations
        if self.use_gradient_checkpointing and self.training:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, t_emb, text_emb, use_reentrant=False)
        else:
            # Standard forward pass
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
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory at the cost of slower training")
    
    # Memory optimization arguments
    parser.add_argument("--teacher_on_cpu", action="store_true",
                        help="Load teacher model on CPU to save GPU memory (slower but uses less VRAM). Deprecated: use --teacher_device_strategy cpu")
    parser.add_argument("--teacher_device_strategy", type=str, default=None,
                        choices=["cpu", "balanced", "gpu0", "auto"],
                        help="Strategy for loading teacher model in distributed training: "
                             "cpu=load on CPU (shared), balanced=distribute across GPUs, "
                             "gpu0=load on GPU 0 only, auto=automatic selection")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision (FP16) training to reduce memory usage")
    parser.add_argument("--teacher_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                        help="Data type for teacher model (float16/bfloat16 use less memory)")

    args = parser.parse_args()

    # 2. Setup Distributed Training & Device
    # This must be done after parsing args but before using rank or device.
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    args.distributed = is_distributed  # Override CLI arg with environment truth

    if args.distributed:
        rank, world_size, local_rank = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
        if is_main_process(rank):
            print(f"Distributed training enabled. Rank: {rank}, World Size: {world_size}, Device: {device}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if is_main_process(rank):
            print(f"Single-process training. Device: {device}")
    
    # 3. Validate and Finalize Arguments
    # Handle backward compatibility: --teacher_on_cpu sets strategy to cpu
    if args.teacher_on_cpu and args.teacher_device_strategy is None:
        args.teacher_device_strategy = "cpu"
    elif args.teacher_on_cpu and args.teacher_device_strategy != "cpu":
        if is_main_process(rank):
            print("Warning: Both --teacher_on_cpu and --teacher_device_strategy specified. Using --teacher_device_strategy.")

    if args.distributed:
        # Set default strategy for distributed training if not specified
        if args.teacher_device_strategy is None:
            args.teacher_device_strategy = "auto"
        
        # Validate teacher device strategy
        num_gpus = torch.cuda.device_count()
        if args.teacher_device_strategy == "auto":
            # Auto-select best strategy based on available resources
            if num_gpus >= 2:
                args.teacher_device_strategy = "balanced"
                if is_main_process(rank):
                    print(f"Auto-selected teacher_device_strategy: balanced (detected {num_gpus} GPUs)")
            else:
                args.teacher_device_strategy = "cpu"
                if is_main_process(rank):
                    print(f"Auto-selected teacher_device_strategy: cpu (only {num_gpus} GPU available)")

        # Print selected strategy
        if is_main_process(rank):
            print("=" * 80)
            print(f"Distributed Training Configuration")
            print("=" * 80)
            print(f"Teacher device strategy: {args.teacher_device_strategy}")
            if args.teacher_device_strategy == "cpu":
                print("  - Teacher will be loaded on CPU (shared by all ranks)")
                print("  - Saves GPU memory, slower teacher inference")
            elif args.teacher_device_strategy == "balanced":
                print("  - Teacher will be distributed across all GPUs")
                print("  - Balanced memory usage, requires inter-GPU communication")
            elif args.teacher_device_strategy == "gpu0":
                print("  - Teacher will be loaded on GPU 0 only")
                print("  - Other ranks will receive teacher outputs via broadcast")
                print("  - NOTE: This requires implementing output broadcasting")
            print("=" * 80)    
    
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
        distributed=args.distributed,
        use_gradient_checkpointing=args.gradient_checkpointing
    )
    
    if is_main_process(rank) and args.gradient_checkpointing:
        print("✓ Gradient checkpointing enabled - trading compute for memory")
    
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

    # 6. Initialize Teacher Model using WAN Text2Video
    # This loads the model using WAN's native components with built-in distribution logic
    if is_main_process(rank) or not args.distributed:
        print(f"Loading teacher model from: {args.teacher_path}")
        print("   Using WAN Text2Video native implementation")
        if args.teacher_device_strategy == "cpu":
            print("   Teacher T5 encoder will be on CPU (t5_cpu=True)")
            print("   Teacher models will offload during generation (offload_model=True)")
        elif args.teacher_device_strategy == "balanced":
            print("   Teacher models will use FSDP for balanced distribution")
        if args.teacher_dtype != "float32":
            print(f"   Teacher model will use {args.teacher_dtype} (via config.param_dtype)")

    # Determine which ranks should load teacher based on strategy
    if args.teacher_device_strategy == "cpu":
        # All ranks can load/share CPU copy
        should_load_teacher = True
    elif args.teacher_device_strategy == "balanced":
        # All ranks participate in balanced loading with FSDP
        should_load_teacher = True
    elif args.teacher_device_strategy == "gpu0":
        # Only rank 0 loads teacher
        should_load_teacher = (rank == 0)
    else:
        # Non-distributed or fallback
        should_load_teacher = (not args.distributed) or (rank == 0)
    
    teacher_wan = None
    if should_load_teacher:
        try:
            if is_main_process(rank):
                print("   Initializing WAN T2V teacher model...")
            
            # Configure WAN based on teacher_device_strategy
            use_t5_cpu = (args.teacher_device_strategy == "cpu")
            use_t5_fsdp = (args.teacher_device_strategy == "balanced" and args.distributed)
            use_dit_fsdp = (args.teacher_device_strategy == "balanced" and args.distributed)
            use_sp = False  # Sequence parallel not needed for distillation
            init_on_cpu = (args.teacher_device_strategy == "cpu")
            
            # Configure dtype based on args
            config = EasyDict(t2v_A14B)
            if args.teacher_dtype == "float16":
                config.param_dtype = torch.float16
                config.t5_dtype = torch.float16
            elif args.teacher_dtype == "bfloat16":
                config.param_dtype = torch.bfloat16
                config.t5_dtype = torch.bfloat16
            
            # Suppress warnings about tied weights in T5/UMT5 models
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*were not used when initializing.*")
                warnings.filterwarnings("ignore", message=".*were newly initialized.*")
                
                teacher_wan = WanT2V(
                    config=config,
                    checkpoint_dir=args.teacher_path,
                    device_id=local_rank,
                    rank=rank,
                    t5_fsdp=use_t5_fsdp,
                    dit_fsdp=use_dit_fsdp,
                    use_sp=use_sp,
                    t5_cpu=use_t5_cpu,
                    init_on_cpu=init_on_cpu,
                    convert_model_dtype=False  # Keep original dtypes
                )
            
            if is_main_process(rank):
                print("✓ Loaded WAN T2V teacher model successfully")
                print(f"   VAE stride: {teacher_wan.vae_stride}")
                print(f"   VAE z_dim: {teacher_wan.vae.model.z_dim}")
                print(f"   T5 on CPU: {use_t5_cpu}")
                print(f"   DiT FSDP: {use_dit_fsdp}")
                if torch.cuda.is_available():
                    print_gpu_memory_summary(rank, "After teacher loading")
                    
        except Exception as e:
            if is_main_process(rank):
                print(f"   Could not load WAN teacher model: {str(e)}")
                import traceback
                traceback.print_exc()
            teacher_wan = None
    else:
        # Ranks that don't load teacher
        if is_main_process(rank):
            print(f"[Rank {rank}] Skipping teacher load - will receive outputs from rank 0")
        teacher_wan = None

    # Exit with error if model couldn't be loaded
    if teacher_wan is None and should_load_teacher:
        if is_main_process(rank):
            print("\n" + "=" * 80, file=sys.stderr)
            print("ERROR: Failed to load WAN teacher model", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(f"Model path: {args.teacher_path}", file=sys.stderr)
            print("\nThe WAN teacher model must be available to run distillation training.", file=sys.stderr)
            print("\nSupported formats:", file=sys.stderr)
            print("\n1. Local checkpoint format:", file=sys.stderr)
            print("   Please ensure the checkpoint directory contains:", file=sys.stderr)
            print("     - models_t5_umt5-xxl-enc-bf16.pth (T5 encoder)", file=sys.stderr)
            print("     - Wan2.1_VAE.pth (VAE)", file=sys.stderr)
            print("     - low_noise_model/ (DiT low noise)", file=sys.stderr)
            print("     - high_noise_model/ (DiT high noise)", file=sys.stderr)
            print("\n2. HuggingFace Diffusers format:", file=sys.stderr)
            print("   The checkpoint directory should contain:", file=sys.stderr)
            print("     - text_encoder/ (T5 encoder)", file=sys.stderr)
            print("     - vae/ (VAE)", file=sys.stderr)
            print("     - transformer/ (DiT low noise)", file=sys.stderr)
            print("     - transformer_2/ (DiT high noise)", file=sys.stderr)
            print("     - tokenizer/ (tokenizer)", file=sys.stderr)
            print("\n   Or use a HuggingFace model ID like:", file=sys.stderr)
            print("     --teacher_path Wan-AI/Wan2.2-T2V-A14B-Diffusers", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
        if args.distributed:
            cleanup_distributed()
        sys.exit(1)

    # Extract references to teacher components for easier access
    teacher_model = None
    teacher_text_encoder = None
    teacher_vae = None
    
    if teacher_wan is not None:
        # Access the low_noise_model as the main teacher (we'll handle boundary in training)
        teacher_model = teacher_wan.low_noise_model
        teacher_text_encoder = teacher_wan.text_encoder
        teacher_vae = teacher_wan.vae
        
        if is_main_process(rank):
            print("--- Teacher Model (WAN) Config ---")
            print(f"VAE latent channels (z_dim): {teacher_vae.model.z_dim}")
            print(f"VAE stride: {teacher_wan.vae_stride}")
            print(f"DiT patch size: {teacher_wan.patch_size}")
            print(f"T5 text length: {teacher_wan.config.text_len}")
            print(f"Param dtype: {teacher_wan.param_dtype}")

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
    
    # MEMORY OPTIMIZATION: Pre-create projection layer if needed
    # This avoids recreating it every batch which causes memory leaks
    # Only create on ranks that have the teacher model
    proj_layer = None
    teacher_device = None
    teacher_dtype = None
    # Get VAE z_dim from teacher VAE config if available
    vae_z_dim = teacher_vae.model.z_dim if teacher_vae is not None else 16
    
    if teacher_model is not None:
        # Use VAE z_dim (latent channels) instead of transformer in_channels
        student_channels = student_config["num_channels"]
        
        if student_channels != vae_z_dim:
            if is_main_process(rank):
                print(f"Creating projection layer: {student_channels} -> {vae_z_dim} channels (VAE z_dim)")
            proj_layer = nn.Conv3d(
                in_channels=student_channels,
                out_channels=vae_z_dim,
                kernel_size=1
            )
            # Initialize weights to zeros
            nn.init.zeros_(proj_layer.weight)
            if proj_layer.bias is not None:
                nn.init.zeros_(proj_layer.bias)
            
            # Move to teacher device/dtype and freeze parameters
            teacher_device = next(teacher_model.parameters()).device
            teacher_dtype = next(teacher_model.parameters()).dtype
            proj_layer = proj_layer.to(device=teacher_device, dtype=teacher_dtype)
            for param in proj_layer.parameters():
                param.requires_grad = False
            if is_main_process(rank):
                print(f"✓ Projection layer created on {teacher_device} with dtype {teacher_dtype}")
        
        # Cache teacher device and dtype to avoid repeated parameter iteration
        teacher_device = next(teacher_model.parameters()).device
        teacher_dtype = next(teacher_model.parameters()).dtype
        if is_main_process(rank):
            print(f"Teacher model device: {teacher_device}, dtype: {teacher_dtype}")
    else:
        # Ranks without teacher model - set defaults
        teacher_device = device
        teacher_dtype = torch.float32


    # 9. Training Loop
    student_model.train()  # Student needs to be in train mode
    global_step = 0

    if is_main_process(rank):
        print("Starting distillation training...")
        # Print initial memory usage
        if torch.cuda.is_available():
            print_gpu_memory_summary(rank, "Initial")

    # Fixed: Use args.num_epochs
    for epoch in range(args.num_epochs):
        # Set epoch for DistributedSampler to ensure different shuffling each epoch
        if args.distributed:
            sampler.set_epoch(epoch)
            
        for batch in dataloader:
            try:
                # --- DATA PREPARATION ---
                # batch is a list of prompt strings
                prompts = batch

                # Encode the prompts using WAN's T5 text encoder
                if teacher_text_encoder is not None:
                    # Use WAN's T5EncoderModel which handles tokenization internally
                    with torch.no_grad():
                        # WAN T5 encoder returns a list of embeddings for each prompt
                        # Shape: list of [seq_len, 4096] tensors (one per prompt)
                        if teacher_wan.t5_cpu:
                            # T5 is on CPU, encode there and move to device
                            text_embeddings_list = teacher_text_encoder(prompts, torch.device('cpu'))
                            text_embeddings_list = [t.to(device) for t in text_embeddings_list]
                        else:
                            # T5 is on GPU, encode directly
                            text_embeddings_list = teacher_text_encoder(prompts, teacher_device)
                        
                        # Stack and pad to create batch tensor
                        # Find max length in batch
                        max_len = max(t.shape[0] for t in text_embeddings_list)
                        batch_size = len(text_embeddings_list)
                        embed_dim = text_embeddings_list[0].shape[1]
                        
                        # Create padded tensor
                        # Note: Zero-padding is appropriate here as the WAN model uses attention masks
                        # and the padded positions will be ignored during cross-attention
                        text_embeddings = torch.zeros(
                            batch_size, max_len, embed_dim,
                            dtype=text_embeddings_list[0].dtype,
                            device=device
                        )
                        for i, emb in enumerate(text_embeddings_list):
                            text_embeddings[i, :emb.shape[0], :] = emb
                else:
                    raise ValueError("No text encoder found in teacher model")

                # Generate random timesteps and latents for the step
                # Use VAE-compatible shape: [batch_size, z_dim, H, W] for 1-frame
                # For 1-frame video generation, we don't need temporal dimension in latents
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
                    # 1. Prepare latents for teacher (WAN model expects list of [C, F, H, W] tensors)
                    # For 1-frame image generation, add temporal dimension
                    if latents.dim() == 4:
                        # [B, C, H, W] -> list of [C, 1, H, W]
                        teacher_latents_list = [latents[i].unsqueeze(1) for i in range(latents.shape[0])]
                    else:
                        # Already has temporal dim
                        teacher_latents_list = [latents[i] for i in range(latents.shape[0])]
                    
                    # 2. Apply projection layer if needed (to match VAE z_dim)
                    # Batch the projection operation for efficiency
                    if proj_layer is not None:
                        # Stack to [B, C, F, H, W], apply projection, then unstack
                        teacher_latents_batch = torch.stack(teacher_latents_list)
                        teacher_latents_batch = proj_layer(teacher_latents_batch)
                        teacher_latents_list = [teacher_latents_batch[i] for i in range(teacher_latents_batch.shape[0])]
                        del teacher_latents_batch
                    
                    # 3. Move to teacher device and dtype
                    teacher_latents_list = [t.to(device=teacher_device, dtype=teacher_dtype) for t in teacher_latents_list]
                    
                    # 4. Prepare text embeddings (WAN expects list of [L, C] tensors)
                    # text_embeddings is [B, L, C], convert to list
                    text_embeddings_teacher = [text_embeddings[i] for i in range(text_embeddings.shape[0])]
                    text_embeddings_teacher = [t.to(device=teacher_device, dtype=teacher_dtype) for t in text_embeddings_teacher]
                    
                    # 5. Prepare timesteps
                    timesteps_teacher = timesteps.to(teacher_device)
                    
                    # 6. Calculate seq_len for positional encoding
                    # Extract dimensions directly from first latent tensor
                    C, F, H, W = teacher_latents_list[0].shape
                    patch_size = teacher_wan.patch_size
                    seq_len = (F // patch_size[0]) * (H // patch_size[1]) * (W // patch_size[2])
                    
                    # Expand timesteps to match the teacher's expectation for a 2D tensor.
                    # The teacher model has an issue handling 1D timestep tensors, so we
                    # expand it here to [batch_size, seq_len] to avoid the error.
                    if timesteps_teacher.dim() == 1:
                        timesteps_teacher = timesteps_teacher.unsqueeze(1).expand(-1, seq_len)

                    # 7. Pass through teacher model
                    # Use autocast for mixed precision to handle dtype mismatches inside the teacher model.
                    # The teacher model internally uses float32 for some operations (like time embeddings)
                    # which can clash with the model's bfloat16/float16 weights if not handled.
                    with torch.autocast(device_type=teacher_device.type, dtype=teacher_dtype):
                        teacher_output_list = teacher_model(
                            x=teacher_latents_list,
                            t=timesteps_teacher,
                            context=text_embeddings_teacher,
                            seq_len=seq_len,
                        )
                    
                    # 8. Convert output list back to batch tensor and remove temporal dim if 1-frame
                    # Output is list of [C, F, H, W], stack to [B, C, F, H, W]
                    teacher_output = torch.stack(teacher_output_list)
                    if teacher_output.shape[2] == 1:
                        # Remove temporal dimension for 1-frame: [B, C, 1, H, W] -> [B, C, H, W]
                        teacher_output = teacher_output.squeeze(2)
                    
                    # 9. Move back to student device
                    teacher_output = teacher_output.to(device).detach()
                    
                    # Free intermediate tensors
                    del teacher_latents_list, text_embeddings_teacher, timesteps_teacher, teacher_output_list

                # --- STUDENT PASS ---

                # Use original latents for student (not the projected ones used for teacher)
                # Student conv_in expects config['num_channels'], not teacher's in_channels
                student_output = student_model(
                    latent_0=latents,  # Use latents directly, not student_latents reference
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
                
                # Save loss value before deleting the tensor
                loss_value = loss.item()
                
                # MEMORY OPTIMIZATION: Explicitly delete tensors and clear cache periodically
                # Only delete tensors that still exist in scope
                del latents, timesteps, text_embeddings, teacher_output, student_output, loss
                
                # Clear CUDA cache every 100 steps to prevent memory fragmentation
                if global_step % 100 == 0:
                    torch.cuda.empty_cache()

                if global_step % 50 == 0 and is_main_process(rank):
                    print(f"Step {global_step}: Loss = {loss_value}")
                    if torch.cuda.is_available() and global_step % 200 == 0:
                        print_gpu_memory_summary(rank, f"Step {global_step}")
            
            except torch.cuda.OutOfMemoryError as e:
                # Print error message from all ranks to ensure visibility
                print("=" * 80, file=sys.stderr)
                print(f"[Rank {rank}] ERROR: CUDA Out of Memory!", file=sys.stderr)
                print("=" * 80, file=sys.stderr)
                print(file=sys.stderr)
                if torch.cuda.is_available():
                    print_gpu_memory_summary(rank, "Current")
                    print(file=sys.stderr)
                
                if is_main_process(rank):
                    print("The model or batch size is too large for available GPU memory.", file=sys.stderr)
                    print(file=sys.stderr)
                    print("Memory-saving solutions (try in order):", file=sys.stderr)
                    print(f"  1. Load teacher on CPU (recommended - frees ~120GB GPU memory):", file=sys.stderr)
                    print(f"     Add --teacher_on_cpu flag", file=sys.stderr)
                    print(file=sys.stderr)
                    print(f"  2. Use lower precision for teacher (saves ~50% memory):", file=sys.stderr)
                    print(f"     Add --teacher_dtype float16 or --teacher_dtype bfloat16", file=sys.stderr)
                    print(file=sys.stderr)
                    print(f"  3. Reduce batch size (current: {args.batch_size}):", file=sys.stderr)
                    print(f"     --batch_size {max(1, args.batch_size // 2)}", file=sys.stderr)
                    print(file=sys.stderr)
                    print("  4. Reduce model size in config/student_config.json:", file=sys.stderr)
                    print("     - hidden_size: 512 (from 1024)", file=sys.stderr)
                    print("     - depth: 8 (from 16)", file=sys.stderr)
                    print("     - image_size: 512 (from 1024)", file=sys.stderr)
                    print(file=sys.stderr)
                    print("  5. Enable gradient checkpointing:", file=sys.stderr)
                    print("     Add --gradient_checkpointing flag", file=sys.stderr)
                    print(file=sys.stderr)
                    print("  6. Set environment variable for better memory management:", file=sys.stderr)
                    print("     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", file=sys.stderr)
                    print(file=sys.stderr)
                    print("Example command with memory optimizations:", file=sys.stderr)
                    print("  torchrun --nproc_per_node=1 train_distillation.py \\", file=sys.stderr)
                    print("    --teacher_on_cpu --teacher_dtype float16 \\", file=sys.stderr)
                    print("    --batch_size 1 --distributed [other args...]", file=sys.stderr)
                    print(file=sys.stderr)
                    print("=" * 80, file=sys.stderr)
                
                # In distributed mode, synchronize all processes before exiting
                if args.distributed:
                    try:
                        dist.barrier()
                    except Exception:
                        pass  # Barrier might fail if some processes already crashed
                    cleanup_distributed()
                
                # Exit with error code
                sys.exit(1)
            
            except Exception as e:
                # Print error from all ranks for visibility in distributed training
                print(f"[Rank {rank}] ERROR during training step: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                
                # In distributed mode, synchronize all processes before exiting
                if args.distributed:
                    try:
                        dist.barrier()
                    except Exception:
                        pass  # Barrier might fail if some processes already crashed
                    cleanup_distributed()
                
                # Exit with error code instead of raising to avoid confusing errors
                sys.exit(1)

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
