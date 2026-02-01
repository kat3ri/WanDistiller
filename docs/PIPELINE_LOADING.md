# Pipeline Loading Performance Issues

## Issue: Loading Hangs at 83% (5/6 Components)

When loading the Wan2.2-T2V-A14B-Diffusers model, you may see:

```
Loading pipeline components...: 83%|████████████████████| 5/6 [00:03<00:00, 1.50it/s]
```

And then it appears to hang or take a very long time.

## Why This Happens

### Pipeline Components

The Wan2.2-T2V model has 6 main components:

1. **Text Encoder (UMT5)** - Encodes text prompts (~1-2GB)
2. **Tokenizer** - Processes text (small, fast)
3. **Transformer** - Main diffusion model (~10-12GB, largest)
4. **VAE** - Video Autoencoder for latent space (~1-2GB)
5. **Scheduler** - Controls diffusion process (small, fast)
6. **Feature Extractor/Processor** - Additional processing (varies)

### Why the 6th Component is Slow

The loading process is sequential, and by the 83% mark:
- The first 5 components have loaded successfully
- The 6th component (often VAE or processor) is loading

**Key Problems:**
1. **Large Model Size**: The complete pipeline is ~14B parameters
2. **CPU Loading**: On CPU, loading float32 weights is extremely slow
3. **Memory Bandwidth**: Moving large tensors to GPU takes time
4. **Default dtype**: Without specifying dtype, Diffusers loads in float32

## Solution Implemented

The code now uses optimized loading:

```python
# GPU: Use float16 for 2x speed and 50% memory reduction
if torch.cuda.is_available():
    load_dtype = torch.float16
    variant = "fp16"  # Use pre-converted fp16 weights
else:
    # CPU: Use bfloat16 if available (better than float32)
    load_dtype = torch.bfloat16  # or torch.float32 as fallback

teacher_pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=load_dtype,
    variant=variant,  # Uses optimized weights when available
    local_files_only=True
)
```

### Benefits

| Setting | Load Speed | Memory | Quality |
|---------|------------|--------|---------|
| **float32 (default)** | Slowest | Highest | Best |
| **float16 (GPU)** | 2x faster | 50% less | Excellent |
| **bfloat16 (CPU)** | 1.5x faster | 50% less | Excellent |

### Additional Optimizations

1. **Separated loading and device transfer**:
   ```python
   # Load to CPU first
   teacher_pipe = DiffusionPipeline.from_pretrained(...)
   # Then move to GPU
   teacher_pipe.to(device)
   ```

2. **Progress messages**:
   - "Loading pipeline components (this may take a few minutes)..."
   - "Moving pipeline to {device}..."

3. **Uses variant="fp16"** on GPU to load pre-converted weights

## Expected Loading Times

### On GPU (CUDA)
- With fp16: **30 seconds - 2 minutes**
- With float32: **2-5 minutes**

### On CPU
- With bfloat16: **5-10 minutes**
- With float32: **10-20 minutes** (or longer on slower CPUs)

### Memory Requirements

- **GPU**: 16GB+ VRAM recommended for fp16
- **CPU**: 32GB+ RAM recommended
- Smaller models can work with less

## Verification

To check if loading is progressing (not hung):

### Windows (PowerShell)
```powershell
Get-Process python | Select-Object CPU, WorkingSet
```

### Linux/Mac
```bash
top -p $(pgrep -f train_distillation.py)
# or
htop
```

Look for:
- CPU usage (should be 10-50%)
- Memory gradually increasing
- Process not in "D" state (disk wait)

## What To Do If It Still Hangs

### 1. Check Available Resources
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 2. Use Smaller Model (for testing)
Try with a smaller model first to verify your setup:
```bash
python train_distillation.py \
    --teacher_path "stabilityai/stable-diffusion-2-1" \
    --student_config config/student_config_small.json \
    ...
```

### 3. Monitor Resource Usage
Open a separate terminal and run:
```bash
# GPU usage
nvidia-smi -l 1  # Updates every second

# CPU/RAM usage
htop
```

### 4. Check Disk Space
The model cache requires significant space:
```bash
# Check cache location (usually ~/.cache/huggingface)
du -sh ~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
```

Ensure you have at least 30-40GB free space.

### 5. Increase Timeout (if needed)
If running with a wrapper script that has timeouts, increase them:
```bash
timeout 1800 python train_distillation.py ...  # 30 minutes
```

## Common Mistakes

❌ **Don't do this:**
```python
# Loading without dtype specification (very slow!)
pipe = DiffusionPipeline.from_pretrained(model_path)
```

✅ **Do this instead:**
```python
# Load with optimized dtype
pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # On GPU
    variant="fp16"
)
```

## Debugging Steps

If loading truly hangs (no progress for 10+ minutes):

1. **Check error logs**: Look for Python exceptions or OOM errors
2. **Try with low_cpu_mem_usage**:
   ```python
   pipe = DiffusionPipeline.from_pretrained(
       model_path,
       torch_dtype=torch.float16,
       low_cpu_mem_usage=True  # Slower but uses less memory
   )
   ```
3. **Clear cache and retry**:
   ```bash
   rm -rf ~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
   ```

## Summary

The 83% hang is usually just slow loading of the final large component. The updated code now:
- ✅ Uses optimized dtype for faster loading
- ✅ Uses fp16 variant on GPU
- ✅ Shows clear progress messages
- ✅ Separates loading from device transfer

**Be patient** - loading a 14B parameter model takes time, especially on CPU!
