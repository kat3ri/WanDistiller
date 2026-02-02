# Model Loading Strategies Guide

This document explains various strategies for loading large models in WanDistiller, including CPU loading, GPU sharding, model offloading, and other memory optimization techniques.

## Overview

WanDistiller supports multiple strategies for loading the teacher model to accommodate different hardware configurations:

1. **CPU Loading** - Load entire model on CPU (saves GPU memory)
2. **GPU Loading** - Load entire model on GPU (fastest inference)
3. **Balanced Loading** - Automatically distribute model across available devices
4. **Sequential Loading** - Place model layers sequentially across devices
5. **Custom Sharding** - Manually specify which layers go on which device

## Strategy 1: CPU Loading (Currently Implemented)

Load the entire teacher model on CPU to save GPU memory. This is useful when you have limited GPU memory or want to reserve GPU for the student model.

### Usage

```bash
python train_distillation.py \
    --teacher_on_cpu \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i"
```

### How It Works

- Uses `low_cpu_mem_usage=True` for efficient CPU memory usage
- Teacher model stays on CPU during training
- Student model can be on GPU for faster training
- Inference is slower but saves significant GPU memory (~120GB)

### Pros/Cons

✅ **Pros:**
- Saves ~120GB GPU memory
- Works on systems with limited GPU resources
- Student model can still use GPU

❌ **Cons:**
- Slower teacher inference (CPU is slower than GPU)
- Requires sufficient RAM (~140GB+ for large models)
- Data transfer between CPU and GPU adds overhead

## Strategy 2: GPU Loading (Default)

Load the entire model on GPU for fastest performance.

### Usage

```bash
# No special flags needed - this is the default
python train_distillation.py \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i"
```

### How It Works

- Uses `device_map={"": device}` to place entire model on specific GPU
- Both teacher and student on same GPU
- No CPU-GPU data transfer overhead

### Pros/Cons

✅ **Pros:**
- Fastest inference and training
- No CPU-GPU transfer overhead
- Simple configuration

❌ **Cons:**
- Requires large GPU memory (~140GB+ for Wan 2.2)
- May not work on consumer GPUs

## Strategy 3: Balanced Loading (Multi-GPU) ✅ IMPLEMENTED

Automatically distribute the model across multiple GPUs using the "balanced" device_map strategy.

### Usage

```bash
# Distributed training with balanced teacher loading
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy balanced \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --distributed

# Or use auto-selection (selects balanced with 2+ GPUs)
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy auto \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --distributed
```

### How It Works

- Accelerate library automatically distributes layers across GPUs
- Balances memory usage across all available GPUs
- Minimizes inter-GPU communication
- Each rank loads its portion of the teacher model

### Pros/Cons

✅ **Pros:**
- Automatic distribution - no manual configuration
- Works across multiple GPUs
- Better memory utilization than single GPU
- Fast inference (all on GPU)

❌ **Cons:**
- Requires multiple GPUs with adequate memory
- Some inter-GPU communication overhead
- May not be perfectly optimized for all models

## Strategy 4: Sequential Loading (Multi-GPU)

Place model layers sequentially across multiple GPUs (GPU 0 gets first layers, GPU 1 gets next layers, etc.).

### Implementation

```python
# In train_distillation.py load_kwargs section
if args.sequential_teacher:
    load_kwargs["device_map"] = "sequential"
```

### How It Works

- Fills GPU 0 first, then GPU 1, etc.
- Simple layer-by-layer placement
- Predictable memory distribution

### Pros/Cons

✅ **Pros:**
- Simple and predictable
- Works with any number of GPUs
- Minimizes complexity

❌ **Cons:**
- May not balance memory optimally
- GPU 0 typically has more load
- Not as efficient as "balanced" strategy

## Strategy 5: Custom Device Mapping (Advanced)

Manually specify which parts of the model go on which device for fine-grained control.

### Implementation

```python
# Custom device map example
custom_device_map = {
    "text_encoder": "cpu",           # Text encoder on CPU
    "unet": 0,                       # UNet on GPU 0
    "vae": 1,                        # VAE on GPU 1
    "transformer": {                 # Transformer layers split
        "layer.0": 0,
        "layer.1": 0,
        "layer.2": 1,
        "layer.3": 1,
    }
}

load_kwargs["device_map"] = custom_device_map
```

### How It Works

- Provides explicit control over component placement
- Can mix CPU and multiple GPUs
- Optimized for specific use cases

### Pros/Cons

✅ **Pros:**
- Maximum flexibility
- Can optimize for specific hardware
- Mix CPU and GPU usage

❌ **Cons:**
- Requires detailed model knowledge
- Complex to configure
- Manual tuning needed for each model

## Strategy 6: Model Offloading (CPU + GPU)

Keep model on CPU and temporarily move parts to GPU as needed.

### Implementation

This requires using Accelerate's `cpu_offload` functionality:

```python
from accelerate import cpu_offload

# Load on CPU
teacher_pipe = DiffusionPipeline.from_pretrained(
    args.teacher_path,
    low_cpu_mem_usage=True
)

# Setup offloading - move to GPU only during forward pass
teacher_pipe = cpu_offload(teacher_pipe, execution_device=device)
```

### How It Works

- Model weights stay on CPU
- During inference, only active layers move to GPU
- Automatically manages CPU-GPU transfers

### Pros/Cons

✅ **Pros:**
- Uses minimal GPU memory
- Works with any GPU size
- Automatic management of transfers

❌ **Cons:**
- Slower than full GPU (due to transfer overhead)
- Requires CPU RAM for full model
- More complex than pure CPU or GPU

## Strategy 7: Model Sharding with Disk Offload

For extremely large models, shard weights and load from disk as needed.

### Implementation

```python
# Enable disk offloading
load_kwargs["offload_folder"] = "offload_weights"
load_kwargs["offload_state_dict"] = True
load_kwargs["device_map"] = "auto"
```

### How It Works

- Saves model shards to disk
- Loads shards on-demand during inference
- Minimizes memory footprint

### Pros/Cons

✅ **Pros:**
- Works with very limited RAM/VRAM
- Can handle models larger than available memory
- No code changes needed for different model sizes

❌ **Cons:**
- Significantly slower (disk I/O bottleneck)
- Requires fast SSD for reasonable performance
- Complex setup

## Recommended Configurations

### Configuration 1: Single GPU (24GB VRAM)
```bash
python train_distillation.py \
    --teacher_on_cpu \
    --teacher_dtype float16 \
    --batch_size 1 \
    ...
```
- Teacher on CPU with FP16
- Student on GPU
- Small batch size

### Configuration 2: Single GPU (48GB+ VRAM)
```bash
python train_distillation.py \
    --teacher_dtype float16 \
    --batch_size 2 \
    ...
```
- Both models on GPU
- FP16 to save memory
- Moderate batch size

### Configuration 3: Multi-GPU (2x24GB)
```bash
# To be implemented: Add --balanced_teacher flag
python train_distillation.py \
    --balanced_teacher \
    --teacher_dtype float16 \
    --batch_size 2 \
    ...
```
- Teacher balanced across GPUs
- Student on GPU 0
- Better memory utilization

### Configuration 4: Multi-GPU with Distributed Training (4x40GB)
```bash
torchrun --nproc_per_node=4 train_distillation.py \
    --distributed \
    --teacher_dtype float16 \
    --batch_size 4 \
    ...
```
- Each process has its own teacher (can be balanced)
- Students distributed across processes
- Maximum training speed

### Configuration 5: CPU-Only System
```bash
python train_distillation.py \
    --teacher_on_cpu \
    --batch_size 1 \
    ...
```
- Both models on CPU
- Requires ~140GB+ RAM
- Very slow but works without GPU

## Current Implementation Status

### ✅ Implemented
- [x] CPU Loading (`--teacher_device_strategy cpu` or legacy `--teacher_on_cpu`)
- [x] GPU Loading (default for non-distributed)
- [x] Balanced multi-GPU loading (`--teacher_device_strategy balanced`)
- [x] GPU0 single GPU loading (`--teacher_device_strategy gpu0`)
- [x] Automatic strategy selection (`--teacher_device_strategy auto`)
- [x] FP16/BF16 precision reduction (`--teacher_dtype`)
- [x] Low memory usage optimization

### 🚧 To Be Implemented
- [ ] Sequential multi-GPU loading (`--sequential` strategy)
- [ ] Custom device mapping (dict-based configuration)
- [ ] Model offloading with cpu_offload (on-demand loading)
- [ ] Disk-based sharding (for extremely large models)
- [ ] Teacher output broadcasting for gpu0 strategy

## Using Implemented Strategies

### CPU Loading
```bash
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy cpu \
    --distributed \
    ...
```

### Balanced Loading
```bash
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy balanced \
    --distributed \
    ...
```

### Auto Selection
```bash
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy auto \
    --distributed \
    ...
```

## Adding New Loading Strategies

To add a new loading strategy:

1. **Add to choices in argument parser**:
```python
parser.add_argument("--teacher_device_strategy", type=str, default=None,
                    choices=["cpu", "balanced", "gpu0", "auto", "sequential"],  # Add new strategy
                    help="Strategy for loading teacher model")
```

2. **Update load_kwargs Logic**:
```python
if args.teacher_device_strategy == "cpu":
    load_kwargs["low_cpu_mem_usage"] = True
elif args.teacher_device_strategy == "balanced":
    load_kwargs["device_map"] = "balanced"
elif args.teacher_device_strategy == "sequential":
    load_kwargs["device_map"] = "sequential"  # New strategy
# ... other strategies
```

3. **Update should_load_teacher Logic** (if needed):
```python
if args.teacher_device_strategy == "sequential":
    should_load_teacher = True  # All ranks participate
```

4. **Test the Strategy**:
```python
# Add test in test_cpu_loading.py or create new test file
def test_balanced_loading_kwargs():
    # Test implementation
    pass
```

## Performance Comparison

| Strategy | GPU Memory | Training Speed | Setup Complexity |
|----------|-----------|----------------|------------------|
| CPU Loading | 0-20GB | Slow (1x) | Simple |
| GPU Loading | 140GB+ | Fast (10x) | Simple |
| Balanced (2 GPU) | 70GB each | Fast (8x) | Medium |
| Sequential (2 GPU) | 70GB each | Fast (7x) | Medium |
| CPU Offload | 10-30GB | Medium (3x) | Complex |
| Disk Sharding | 5-15GB | Very Slow (0.3x) | Complex |

*Speeds are approximate relative to CPU loading baseline*

## Troubleshooting

### Error: "cpu not supported. Supported strategies are: balanced, cuda"

**Problem:** Trying to use `device_map="cpu"` which is not a valid strategy.

**Solution:** Use `low_cpu_mem_usage=True` instead of `device_map="cpu"`:
```python
load_kwargs["low_cpu_mem_usage"] = True
# Not: load_kwargs["device_map"] = "cpu"
```

### Error: "CUDA out of memory"

**Solutions:**
1. Use `--teacher_on_cpu` to offload teacher to CPU
2. Reduce precision with `--teacher_dtype float16`
3. Reduce batch size with `--batch_size 1`
4. Use balanced loading across multiple GPUs
5. Enable gradient checkpointing with `--gradient_checkpointing`

### Error: "Not enough RAM"

**Solutions:**
1. Use GPU loading instead of CPU loading
2. Use disk-based sharding
3. Reduce model size in config
4. Use model offloading instead of full CPU loading

## References

- [Hugging Face Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [Diffusers Model Loading Guide](https://huggingface.co/docs/diffusers/using-diffusers/loading)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)

## See Also

- [README.md](README.md) - Main documentation
- [docs/MULTI_GPU.md](docs/MULTI_GPU.md) - Multi-GPU training guide
- [MEMORY_OPTIMIZATIONS.md](MEMORY_OPTIMIZATIONS.md) - Memory optimization strategies
