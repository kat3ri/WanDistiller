# WanDistiller - Text-to-Image Model Distillation

A comprehensive framework to distill the **Wan 2.2 Video Model** into a lightweight **2D Image-Only Model** (Wan-Lite) for high-quality static image generation.

## 🎯 Overview

**WanDistiller** converts a 3D video generation model into a 2D image generation model by:
- **Stripping temporal/motion components** from the teacher (Wan 2.2)
- **Distilling knowledge** through noise prediction matching
- **Using projection mapping** to convert 3D video weights to 2D image weights
- **Creating a smaller, faster model** optimized for static images

### Architecture

- **Teacher Model**: Wan 2.2 (3D Video Generation Model) - holds image quality knowledge
- **Student Model**: Wan-Lite (2D Image Model) - smaller and faster, for static images only
- **Method**: Knowledge distillation via MSE loss on noise predictions
- **Result**: High-fidelity text-to-image generation without motion artifacts

## 📋 Prerequisites

- Python 3.9+
- PyTorch 2.0+
- 8GB+ RAM (16GB+ recommended for full-size training)
- GPU optional but recommended for production training

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Production Test

**Verify the pipeline works with mock data (recommended first step):**

```bash
python run_production_test.py
```

This will:
- ✅ Load 55 prompts from `data/static_prompts.txt`
- ✅ Initialize a small test model (256 hidden size, 4 layers)
- ✅ Run 1 epoch of training with mocked teacher
- ✅ Save the trained model
- ✅ Verify everything works correctly

**Expected output:**
```
================================================================================
✓ Production Test PASSED
================================================================================
```

See [TESTING.md](TESTING.md) for detailed testing documentation.

### 3. Train with Real Data

Once the test passes, train with actual model weights:

```bash
# Single GPU training
python train_distillation.py \
    --teacher_path "Wan-AI/Wan2.2-T2V-A14B-Diffusers" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --num_epochs 100 \
    --batch_size 4 \
    --lr 1e-5

# Multi-GPU training (recommended for faster training)
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_path "Wan-AI/Wan2.2-T2V-A14B-Diffusers" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --num_epochs 100 \
    --batch_size 4 \
    --lr 1e-5 \
    --distributed
```

> **⚠️ Important:** When using `torchrun`, do NOT include `python` before the script name. 
> The command `torchrun python train_distillation.py` will fail because `torchrun` already invokes Python internally.
> 
> **Correct:** `torchrun --nproc_per_node=4 train_distillation.py ...`  
> **Wrong:** `torchrun --nproc_per_node=4 python train_distillation.py ...`

See [docs/MULTI_GPU.md](docs/MULTI_GPU.md) for detailed multi-GPU training guide.

### Sample Image Generation During Training

You can optionally generate sample images from the student model during training to monitor progress:

```bash
# Enable sample generation (once per epoch by default)
python train_distillation.py \
    --teacher_path "Wan-AI/Wan2.2-T2V-A14B-Diffusers" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --num_epochs 100 \
    --batch_size 4 \
    --lr 1e-5 \
    --save_samples \
    --sample_prompts "A serene mountain landscape at sunset" "A futuristic city with neon lights"

# Custom sample interval (generate every 5 epochs)
python train_distillation.py \
    --teacher_path "Wan-AI/Wan2.2-T2V-A14B-Diffusers" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --num_epochs 100 \
    --batch_size 4 \
    --lr 1e-5 \
    --save_samples \
    --sample_interval 5 \
    --sample_dir "./outputs/wan_t2i/my_samples"
```

**Sample Generation Options:**
- `--save_samples`: Enable sample image generation during training
- `--sample_prompts`: Text prompts to use for generation (default: 2 example prompts)
- `--sample_interval`: Generate samples every N epochs (default: 1, i.e., once per epoch)
- `--sample_dir`: Directory to save samples (default: `{output_dir}/samples`)

Samples are saved as `epoch_NNNN_sample_MM.png` in the sample directory, where NNNN is the epoch number and MM is the sample index.

## 📁 Project Structure

```
WanDistiller/
├── README.md                    # This file
├── TESTING.md                   # Testing documentation
├── requirements.txt             # Python dependencies
├── run_production_test.py       # Production test script
├── train_distillation.py        # Main training script (supports multi-GPU)
├── projection_mapper.py         # 3D-to-2D weight projection
├── main.py                      # Simple entry point
├── config/
│   └── student_config.json      # Student model architecture
├── data/
│   └── static_prompts.txt       # Training prompts (55 samples)
└── docs/
    ├── MULTI_GPU.md             # Multi-GPU training guide
    ├── UMT5_WEIGHT_LOADING.md   # UMT5 weight loading info
    └── PIPELINE_LOADING.md      # Pipeline loading troubleshooting
```

## 🔑 Key Features

### 2D Image-Only Architecture
The student model is **purely spatial (2D)** with:
- ✅ No temporal dimensions
- ✅ No video/motion components
- ✅ Optimized for static image generation
- ✅ Conv2D layers (not Conv3D)
- ✅ 2D spatial attention only

### Multi-GPU Training Support
Accelerate your training with multiple GPUs:
- ✅ **DataParallel** - Simple multi-GPU training on a single machine
- ✅ **DistributedDataParallel** - Advanced multi-GPU with better performance
- ✅ Automatic batch distribution across GPUs
- ✅ Support for multi-machine training
- ✅ See [docs/MULTI_GPU.md](docs/MULTI_GPU.md) for details

### Intelligent Weight Projection
The `projection_mapper.py` handles:
- Converting 3D video model weights → 2D image model weights
- Handling dimension mismatches between teacher and student
- Intelligently initializing projection layers
- Preserving learned features while adapting architecture

### Rich Mock Data
`data/static_prompts.txt` includes 55 diverse prompts:
- 🎨 Portrait photography
- 🏔️ Landscape scenes
- 🤖 Sci-fi and cyberpunk themes
- 🏰 Fantasy settings
- 🌆 Architecture
- 🌸 Nature scenes
- And more...

## 📊 Model Configuration

Edit `config/student_config.json` to adjust the model size:

```json
{
  "model_type": "WanLiteStudent",
  "hidden_size": 1024,        // Embedding dimension
  "depth": 16,                // Number of transformer layers
  "num_heads": 16,            // Attention heads
  "num_channels": 4,          // Latent channels
  "image_size": 1024,         // Target image size
  "patch_size": 16,           // Patch size for tokenization
  "text_max_length": 77,      // Max text tokens
  "text_encoder_output_dim": 4096,  // Text encoder dimension
  "projection_factor": 1.0    // Weight projection scale
}
```

**For testing/debugging**, use smaller values:
- `hidden_size: 256`
- `depth: 4`
- `image_size: 256`

## 🧪 Testing

### Quick Test (Mock Teacher)
```bash
python run_production_test.py --num-epochs 2 --batch-size 2
```

### Full Test with Custom Parameters
```bash
python run_production_test.py \
    --num-epochs 5 \
    --batch-size 4
```

### Test with Real Model (Requires Download)
```bash
python run_production_test.py --use-real-teacher
```

See [TESTING.md](TESTING.md) for complete testing guide.

## 🔬 How It Works

### Distillation Process

1. **Teacher Forward Pass** (3D Video Model)
   - Generates predictions from noisy latents
   - Uses full video model architecture
   - No gradients computed (frozen)

2. **Student Forward Pass** (2D Image Model)
   - Generates predictions from same noisy latents
   - Uses lightweight 2D architecture
   - Gradients computed for training

3. **Loss Calculation**
   ```python
   loss = MSE(student_output, teacher_output)
   ```

4. **Weight Update**
   - Student learns to match teacher's predictions
   - Only student weights are updated
   - Knowledge transferred without motion components

### Weight Projection (3D → 2D)

The `projection_mapper.py` converts teacher weights:

```python
# Teacher: 3D Conv (C_out, C_in, D, H, W)
# Student: 2D Conv (C_out, C_in, H, W)

# Projection handles:
# - Dimension reduction (3D → 2D)
# - Channel adaptation
# - Weight initialization
# - Proper scaling
```

## 📈 Training Tips

1. **Start Small**: Test with small model config first
2. **Monitor Loss**: Loss should decrease over epochs
3. **Use GPU**: Much faster than CPU for full training
4. **Batch Size**: Adjust based on available memory
5. **Learning Rate**: 1e-5 is a good starting point
6. **Add More Prompts**: More diverse prompts = better generalization
7. **Monitor Progress**: Use `--save_samples` to generate sample images during training and visually track improvement

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

If you encounter CUDA Out of Memory errors during distributed training:

```bash
# OPTION 1: Balanced loading (distribute teacher across GPUs)
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy balanced \
    --batch_size 2 \
    --distributed \
    [other args...]

# OPTION 2: Load teacher on CPU (saves ~120GB GPU memory per GPU)
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy cpu \
    --batch_size 2 \
    --distributed \
    [other args...]

# OPTION 3: Load teacher on GPU 0 only (requires output broadcasting - experimental)
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy gpu0 \
    --batch_size 2 \
    --distributed \
    [other args...]

# OPTION 4: Auto-select best strategy (recommended for beginners)
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy auto \
    --batch_size 2 \
    --distributed \
    [other args...]

# Alternative: Use lower precision for teacher (saves ~50% memory)
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy balanced \
    --teacher_dtype float16 \
    --batch_size 2 \
    --distributed \
    [other args...]

# Combine multiple memory optimizations for maximum savings
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_device_strategy cpu \
    --teacher_dtype float16 \
    --batch_size 1 \
    --gradient_checkpointing \
    --distributed \
    [other args...]

# Backward compatibility: --teacher_on_cpu still works
torchrun --nproc_per_node=4 train_distillation.py \
    --teacher_on_cpu \
    --batch_size 2 \
    --distributed \
    [other args...]

# 1. Reduce batch size (most effective for student model)
torchrun --nproc_per_node=2 train_distillation.py \
    --batch_size 1 \
    --distributed \
    [other args...]

# 2. Enable gradient checkpointing (trades compute for memory)
torchrun --nproc_per_node=2 train_distillation.py \
    --batch_size 2 \
    --distributed \
    --gradient_checkpointing \
    [other args...]

# 3. Use fewer GPUs if memory per GPU is limited
torchrun --nproc_per_node=1 train_distillation.py \
    --distributed \
    [other args...]

# 4. Enable memory-efficient allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=2 train_distillation.py --distributed [other args...]

# 5. Reduce model size in config/student_config.json
# Edit these parameters:
# - "hidden_size": 512 (reduce from 1024)
# - "depth": 8 (reduce from 16)
# - "image_size": 512 (reduce from 1024)
```

**Teacher Device Strategies for Distributed Training:**
- `--teacher_device_strategy cpu`: Load teacher on CPU (shared by all ranks), saves GPU memory
- `--teacher_device_strategy balanced`: Distribute teacher across all GPUs, best for multiple GPUs
- `--teacher_device_strategy gpu0`: Load teacher on GPU 0 only (experimental, requires broadcasting)
- `--teacher_device_strategy auto`: Automatically select best strategy (default for distributed)
- `--teacher_on_cpu`: Deprecated but still supported, equivalent to `--teacher_device_strategy cpu`

**Memory-Saving Options:**
- `--teacher_dtype float16`: Use FP16 for teacher model (saves ~50% memory)
- `--teacher_dtype bfloat16`: Use BF16 for teacher model (saves ~50% memory, better numerical stability than FP16)
- `--gradient_checkpointing`: Trade compute for memory in student model

**Memory Usage Tips:**
- **Teacher model is the biggest memory consumer** (~120GB in FP32)
- Use `balanced` strategy to distribute teacher across multiple GPUs
- Use `cpu` strategy if you have limited GPU memory but sufficient RAM
- Larger batch sizes require more memory
- Larger image_size increases memory usage quadratically
- Gradient checkpointing reduces memory by ~40% but slows training by ~20%
- Monitor GPU memory with `nvidia-smi` during training

**Memory Optimizations Applied:**
- Optional CPU offloading for teacher model (saves most memory)
- Optional lower precision (FP16/BF16) for teacher model
- Automatic cleanup of unused teacher pipeline components (VAE, scheduler)
- Projection layers are cached instead of recreated each batch
- Explicit tensor cleanup after each training step
- CUDA cache cleared periodically to prevent fragmentation

### Slow Training
```bash
# Use GPU if available
# Or reduce image_size in config
# Or reduce num_epochs for testing
```

### CPU Loading Error: "cpu not supported. Supported strategies are: balanced, cuda"

**This error has been fixed!** If you encounter this error in an older version:

- **Problem:** The code was incorrectly using `device_map="cpu"`, which is not a valid strategy for the Accelerate library
- **Solution:** Update to the latest version which uses `low_cpu_mem_usage=True` instead
- **Details:** See [docs/MODEL_LOADING_STRATEGIES.md](docs/MODEL_LOADING_STRATEGIES.md) for all available loading strategies

Valid device_map strategies include: `"auto"`, `"balanced"`, `"sequential"`, or custom dict mappings like `{"": device}`. For CPU loading, use `low_cpu_mem_usage=True` without device_map.

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### UMT5 "encoder.embed_tokens.weight MISSING" Warning
This warning is **expected and harmless**. The UMT5 text encoder uses weight tying where `encoder.embed_tokens.weight` references `shared.weight`. The weights are correctly loaded - see [docs/UMT5_WEIGHT_LOADING.md](docs/UMT5_WEIGHT_LOADING.md) for details.

### Pipeline Loading Hangs at 83% (5/6 components)
If the model loading appears to hang at "Loading pipeline components: 83%", this is usually the VAE or final component loading. The updated code now:
- Uses optimized dtype (float16 on GPU, bfloat16/float32 on CPU) for faster loading
- Uses fp16 variant when available on GPU
- Shows progress messages during loading and device transfer

**If it still takes too long**: The Wan2.2 model is very large (~14B parameters). Loading can take 5-10 minutes on CPU or slower systems. Be patient and wait for the process to complete. See [docs/PIPELINE_LOADING.md](docs/PIPELINE_LOADING.md) for detailed troubleshooting.

## 📚 Additional Documentation

- [TESTING.md](TESTING.md) - Complete testing guide
- [docs/MULTI_GPU.md](docs/MULTI_GPU.md) - **Multi-GPU training guide (NEW!)**
- [docs/MODEL_LOADING_STRATEGIES.md](docs/MODEL_LOADING_STRATEGIES.md) - **Model loading strategies: CPU, GPU, sharding, offloading (NEW!)**
- [docs/UMT5_WEIGHT_LOADING.md](docs/UMT5_WEIGHT_LOADING.md) - Understanding UMT5 text encoder weight warnings
- [docs/PIPELINE_LOADING.md](docs/PIPELINE_LOADING.md) - Troubleshooting slow pipeline loading
- [readme.md](readme.md) - Original detailed documentation
- `config/student_config.json` - Model architecture reference

## 🎓 Citation

If you use this framework, please cite:

```bibtex
@software{wandistiller2024,
  title={WanDistiller: Text-to-Image Model Distillation Framework},
  author={WanDistiller Contributors},
  year={2024},
  url={https://github.com/kat3ri/WanDistiller}
}
```

## 📝 License

See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run the production test
4. Submit a pull request

## 🔗 Related Projects

- [Wan 2.2](https://huggingface.co/timbrooks/instruct-wan) - Teacher model
- [Diffusers](https://github.com/huggingface/diffusers) - Core diffusion library
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

**Status**: ✅ Production test passing | 🎯 Ready for training | 📦 55 mock prompts included
