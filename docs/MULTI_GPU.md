# Multi-GPU Training Guide

WanDistiller now supports multi-GPU training to accelerate the distillation process. This guide explains how to use the multi-GPU features.

## Overview

The framework supports two modes of multi-GPU training:

1. **DataParallel (DP)** - Simple multi-GPU training on a single machine
2. **DistributedDataParallel (DDP)** - Advanced multi-GPU training with better performance

## Prerequisites

- Multiple CUDA-capable GPUs
- PyTorch 2.0+ with CUDA support
- NCCL backend (included with PyTorch CUDA builds)

## Usage

### 1. DataParallel (Simple Multi-GPU)

DataParallel is the simplest way to use multiple GPUs on a single machine. It automatically splits batches across available GPUs.

```bash
python train_distillation.py \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --num_epochs 100 \
    --batch_size 4 \
    --lr 1e-5 \
    --multi_gpu
```

**Pros:**
- Easy to use - just add `--multi_gpu` flag
- No additional setup required
- Works on single machine

**Cons:**
- Less efficient than DDP
- Main GPU has higher memory usage
- Communication overhead between GPUs

**Best for:**
- Quick testing with multiple GPUs
- Single-machine setups
- Small to medium models

### 2. DistributedDataParallel (Recommended)

DistributedDataParallel (DDP) is the recommended approach for multi-GPU training. It provides better performance and scalability.

#### Single Machine (Recommended)

Use `torch.distributed.launch` or `torchrun` to start distributed training:

> **⚠️ Important:** When using `torchrun`, do NOT include `python` before the script name.  
> The command `torchrun python train_distillation.py` will fail because `torchrun` already invokes Python internally.  
> **Correct:** `torchrun train_distillation.py ...`  
> **Wrong:** `torchrun python train_distillation.py ...`

```bash
# Using torchrun (PyTorch 2.0+, recommended)
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --num_epochs 100 \
    --batch_size 4 \
    --lr 1e-5 \
    --distributed

# Or using torch.distributed.launch (older PyTorch, deprecated)
python -m torch.distributed.launch --nproc_per_node=4 train_distillation.py \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --num_epochs 100 \
    --batch_size 4 \
    --lr 1e-5 \
    --distributed
```

Where:
- `--nproc_per_node=4` specifies the number of GPUs to use (4 in this example)
- `--distributed` enables DDP mode

#### Multiple Machines

For training across multiple machines, set additional environment variables:

```bash
# On the main node (rank 0)
export MASTER_ADDR=192.168.1.1  # IP of the main node
export MASTER_PORT=29500
export WORLD_SIZE=8  # Total number of processes across all nodes
export RANK=0  # Rank of this node

torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    train_distillation.py \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --num_epochs 100 \
    --batch_size 4 \
    --lr 1e-5 \
    --distributed

# On the second node (rank 1)
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=4  # Starting rank for this node

torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    train_distillation.py \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --num_epochs 100 \
    --batch_size 4 \
    --lr 1e-5 \
    --distributed
```

**Pros:**
- Best performance and scalability
- Efficient GPU utilization
- Supports multiple machines
- Lower memory overhead per GPU

**Cons:**
- Slightly more complex setup
- Requires `torch.distributed.launch` or `torchrun`

**Best for:**
- Production training
- Multi-machine setups
- Large models
- Maximum performance

## Batch Size Considerations

When using multi-GPU training, the effective batch size is:

```
effective_batch_size = batch_size × num_gpus
```

For example:
- Single GPU: `--batch_size 4` → effective batch size = 4
- 4 GPUs with DP/DDP: `--batch_size 4` → effective batch size = 16

**Recommendations:**
- Keep the effective batch size consistent with single-GPU training
- If using 4 GPUs and want effective batch size of 16, use `--batch_size 4`
- Adjust learning rate if changing effective batch size (linear scaling rule: `lr × num_gpus`)

## Performance Tips

1. **Use DistributedDataParallel** - Always prefer DDP over DataParallel for better performance

2. **Adjust batch size** - Larger batch sizes on each GPU improve GPU utilization:
   ```bash
   # 4 GPUs, effective batch size = 32
   --batch_size 8 --distributed
   ```

3. **Enable mixed precision** - Add mixed precision training for faster training (future feature)

4. **Monitor GPU utilization**:
   ```bash
   # In a separate terminal
   watch -n 1 nvidia-smi
   ```

5. **Pin memory** - Already enabled by default for distributed training

## Troubleshooting

### Issue: "NCCL error" or communication timeout

**Solution:**
```bash
# Set environment variables before running
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Replace with your network interface
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
```

### Issue: Out of memory (OOM) errors

**Solutions:**
1. Reduce batch size: `--batch_size 2`
2. Reduce model size in `config/student_config.json`
3. Enable gradient checkpointing (future feature)

### Issue: Slow data loading

**Solution:**
The default `num_workers=0` avoids multiprocessing issues but can be slow. For faster data loading:

```bash
# Increase data loading workers (recommended: 4-8)
python train_distillation.py \
    --num_workers 4 \
    --distributed \
    # ... other args
```

Tips:
1. Ensure data is on fast storage (SSD)
2. Start with `--num_workers 4` and increase if stable
3. Set `num_workers` to number of CPU cores for maximum performance
4. Too many workers can cause memory issues - tune based on your system

### Issue: Processes hang at initialization

**Solution:**
```bash
# Ensure all processes can communicate
# Check firewall settings
# Try setting timeout
export NCCL_TIMEOUT=600  # 10 minutes
```

## Verification

To verify multi-GPU training is working:

1. **Check GPU utilization:**
   ```bash
   watch -n 1 nvidia-smi
   ```
   You should see all GPUs being utilized.

2. **Check training output:**
   - For DDP: You'll see messages like `[Rank 0] Distributed training initialized`
   - For DP: You'll see `Model wrapped with DataParallel on N GPUs`

3. **Compare training speed:**
   - Training should be approximately N× faster with N GPUs (accounting for communication overhead)

## Example: Full Training with 4 GPUs

```bash
# Using DistributedDataParallel (recommended)
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i_4gpu" \
    --num_epochs 100 \
    --batch_size 4 \
    --lr 1e-5 \
    --distributed

# This will use:
# - 4 GPUs
# - Effective batch size: 16 (4 per GPU × 4 GPUs)
# - Distributed data loading with DistributedSampler
# - Synchronized gradient updates across all GPUs
```

## Further Reading

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)

## Summary

| Feature | Single GPU | DataParallel | DistributedDataParallel |
|---------|-----------|--------------|------------------------|
| Ease of Use | ✅ Easy | ✅ Easy | ⚠️ Moderate |
| Performance | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| Multi-Machine | ❌ | ❌ | ✅ |
| Memory Efficiency | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Recommended | Small models | Testing | Production |

**Recommendation:** Use DistributedDataParallel (`--distributed`) for best performance!
