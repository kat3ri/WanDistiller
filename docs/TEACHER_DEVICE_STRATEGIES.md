# Teacher Device Strategy Examples

This document provides practical examples of each teacher device strategy option for distributed training.

## Overview

WanDistiller now supports flexible teacher loading strategies for distributed training:
- **CPU**: Load teacher on CPU (shared across all ranks)
- **Balanced**: Distribute teacher across all GPUs
- **GPU0**: Load teacher on GPU 0 only
- **Auto**: Automatically select best strategy

## Strategy 1: CPU Loading

**Best for:** Limited GPU memory, sufficient RAM

**How it works:** Teacher model is loaded on CPU memory and shared by all training processes.

### Example

```bash
# 4 GPU setup with CPU teacher
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --teacher_device_strategy cpu \
    --distributed \
    --batch_size 2 \
    --num_epochs 10
```

### Memory Distribution
- **GPU 0**: Student model (~10GB)
- **GPU 1**: Student model (~10GB)
- **GPU 2**: Student model (~10GB)
- **GPU 3**: Student model (~10GB)
- **CPU RAM**: Teacher model (~120GB, shared)
- **Total GPU**: 40GB
- **Total RAM**: 120GB

### Pros
✅ Minimal GPU memory usage
✅ Works with consumer GPUs
✅ All ranks can access teacher

### Cons
❌ Slower teacher inference (CPU)
❌ Requires large RAM (~140GB+)
❌ CPU-GPU data transfer overhead

## Strategy 2: Balanced Loading

**Best for:** Multiple GPUs with adequate memory

**How it works:** Accelerate automatically distributes teacher layers across all available GPUs.

### Example

```bash
# 4 GPU setup with balanced teacher
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --teacher_device_strategy balanced \
    --distributed \
    --batch_size 2 \
    --num_epochs 10
```

### Memory Distribution
- **GPU 0**: Student (~10GB) + Teacher shard (~30GB) = 40GB
- **GPU 1**: Student (~10GB) + Teacher shard (~30GB) = 40GB
- **GPU 2**: Student (~10GB) + Teacher shard (~30GB) = 40GB
- **GPU 3**: Student (~10GB) + Teacher shard (~30GB) = 40GB
- **Total GPU**: 160GB (distributed)

### Pros
✅ Fast teacher inference (all on GPU)
✅ Balanced memory across GPUs
✅ No CPU-GPU transfer overhead
✅ Automatic layer distribution

### Cons
❌ Requires 40GB+ per GPU
❌ Inter-GPU communication overhead
❌ More complex setup

## Strategy 3: GPU0 Loading (Experimental)

**Best for:** Special cases where only rank 0 should have teacher

**How it works:** Only GPU 0 loads teacher. Other ranks receive outputs via broadcast.

### Example

```bash
# 4 GPU setup with teacher on GPU 0 only
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --teacher_device_strategy gpu0 \
    --distributed \
    --batch_size 2 \
    --num_epochs 10
```

### Memory Distribution
- **GPU 0**: Student (~10GB) + Teacher (~120GB) = 130GB
- **GPU 1**: Student (~10GB)
- **GPU 2**: Student (~10GB)
- **GPU 3**: Student (~10GB)
- **Total GPU**: 160GB

### Pros
✅ Fast teacher inference (GPU)
✅ Minimal GPU memory on ranks 1-3
✅ No CPU RAM needed

### Cons
❌ GPU 0 needs 130GB+ memory
❌ Requires output broadcasting (experimental)
❌ Imbalanced GPU usage

⚠️ **Note**: This strategy requires implementing teacher output broadcasting. Currently may not work correctly.

## Strategy 4: Auto Selection (Recommended)

**Best for:** Beginners, dynamic environments

**How it works:** Automatically selects the best strategy based on available hardware.
- ≥2 GPUs → Balanced
- 1 GPU → CPU

### Example

```bash
# Let WanDistiller choose the best strategy
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --teacher_device_strategy auto \
    --distributed \
    --batch_size 2 \
    --num_epochs 10

# Or simply omit the flag (auto is default for distributed)
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --distributed \
    --batch_size 2 \
    --num_epochs 10
```

### Selection Logic
```
if num_gpus >= 2:
    strategy = "balanced"
else:
    strategy = "cpu"
```

### Pros
✅ No manual configuration needed
✅ Adapts to available hardware
✅ Good defaults for most cases

### Cons
❌ Less control
❌ May not be optimal for all setups

## Backward Compatibility

The old `--teacher_on_cpu` flag still works:

```bash
# Old way (still supported)
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_on_cpu \
    --distributed \
    ...

# Equivalent to:
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy cpu \
    --distributed \
    ...
```

## Combining with Other Options

### With Lower Precision

```bash
# Balanced + FP16 (saves 50% memory)
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy balanced \
    --teacher_dtype float16 \
    --distributed \
    ...
```

### With Gradient Checkpointing

```bash
# CPU teacher + gradient checkpointing
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy cpu \
    --gradient_checkpointing \
    --distributed \
    ...
```

### Maximum Memory Savings

```bash
# All optimizations combined
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy cpu \
    --teacher_dtype float16 \
    --gradient_checkpointing \
    --batch_size 1 \
    --distributed \
    ...
```

## Hardware Recommendations

### 4x A100 (80GB each)
```bash
# Use balanced for best performance
--teacher_device_strategy balanced
```

### 4x RTX 3090 (24GB each)
```bash
# Use CPU to save GPU memory
--teacher_device_strategy cpu
--teacher_dtype float16
```

### 2x A6000 (48GB each)
```bash
# Use balanced with FP16
--teacher_device_strategy balanced
--teacher_dtype float16
```

### 1x A100 (80GB)
```bash
# Single GPU, use regular training (no distributed)
python train_distillation.py \
    --teacher_dtype float16 \
    ...
```

## Troubleshooting

### Error: "CUDA out of memory"
Try:
1. Switch to `cpu` strategy
2. Use `float16` for teacher
3. Reduce batch size
4. Enable gradient checkpointing

### Error: "balanced strategy not working"
- Ensure all GPUs are visible: `nvidia-smi`
- Check CUDA_VISIBLE_DEVICES is not set
- Verify accelerate is installed

### Slow training with CPU strategy
- Expected behavior (CPU is slower than GPU)
- Consider upgrading to more RAM and faster CPU
- Or use balanced strategy if GPU memory allows

## Performance Comparison

| Strategy | Speed | GPU Memory | RAM | Best For |
|----------|-------|------------|-----|----------|
| CPU | Slow (1x) | Low (~10GB/GPU) | High (~140GB) | Limited GPU memory |
| Balanced | Fast (8-10x) | Medium (~40GB/GPU) | Low | Multiple GPUs |
| GPU0 | Fast (8-10x) | High on GPU0 (~130GB) | Low | Special cases |
| Auto | Varies | Varies | Varies | General use |

## Summary

Choose your strategy based on your hardware:

- **Have 4x 80GB GPUs?** → Use `balanced`
- **Have 4x 24GB GPUs?** → Use `cpu`
- **Not sure?** → Use `auto` (default)
- **Need max speed?** → Use `balanced` with sufficient GPU memory
- **Need min memory?** → Use `cpu` with `float16`
