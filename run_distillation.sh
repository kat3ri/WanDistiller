#!/bin/bash
set -e

# --- Environment Setup ---
echo "--- Setting up environment ---"
VENV_ACTIVATE=""

# This script assumes it is run from the root of the WanDistiller directory.
if [ -f "venv/bin/activate" ]; then
    VENV_ACTIVATE="venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    VENV_ACTIVATE=".venv/bin/activate"
fi

if [ -z "$VENV_ACTIVATE" ]; then
    echo "Error: Virtual environment not found in './venv/' or './.venv/'"
    echo "Please ensure the script is run from the 'WanDistiller' directory."
    exit 1
fi

echo "Activating virtual environment: $VENV_ACTIVATE"
source "$VENV_ACTIVATE"

# --- Configuration ---
# Auto-detect available GPUs.
# In a cluster environment, the scheduler often sets CUDA_VISIBLE_DEVICES.
# We should respect that if it's set. If not, we can try to detect all GPUs.
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Detecting all available GPUs with nvidia-smi."
    if command -v nvidia-smi &> /dev/null; then
        GPU_INDICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
        if [ -n "$GPU_INDICES" ]; then
            export CUDA_VISIBLE_DEVICES=$GPU_INDICES
            echo "Automatically set CUDA_VISIBLE_DEVICES to: $CUDA_VISIBLE_DEVICES"
        fi
    else
        echo "nvidia-smi not found. Relying on PyTorch to detect devices."
    fi
fi

# Now, determine nproc_per_node based on what PyTorch can see.
# This correctly handles cases where CUDA_VISIBLE_DEVICES is set by the scheduler or by the script above.
if python -c 'import torch; exit(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else exit(1)'; then
    NPROC_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
    echo "PyTorch reports $NPROC_PER_NODE available CUDA device(s)."
else
    echo "No CUDA devices found by PyTorch. Setting to 1 process for CPU execution."
    NPROC_PER_NODE=1
fi


# --- Run Training ---
echo "Starting distillation training..."

# Note: The `-- distributed` flag from the original script has been removed as it seems to be a typo.
# `torchrun` handles the distributed setup, so an explicit flag is often not needed for the training script.
# If your script `train_distillation.py` specifically requires it, you can add it back.
torchrun --nproc_per_node=$NPROC_PER_NODE train_distillation.py \
    --teacher_path "./Wan2.2-T2V-A14B" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --num_epochs 20 \
    --batch_size 1 \
    --lr 1e-5 \
    --gradient_checkpointing 
#    --save_samples \
#    --sample_prompts "A mountain landscape" "A city at night" "a woman standing in a field of sunflowers" \
#    --sample_interval 10

echo "Training finished."
