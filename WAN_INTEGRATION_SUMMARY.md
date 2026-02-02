# WAN Integration Summary

## Overview

This document summarizes the integration of WAN's native text2video module into the distillation training pipeline, replacing the custom teacher loading and inference logic.

## Problem Statement

The original implementation used:
1. Custom teacher latent generation logic
2. Custom distribution strategies for CUDA OOM mitigation
3. Diffusers WanPipeline which didn't leverage WAN's built-in optimizations
4. Manual shape conversions between 2D student and 3D teacher

## Solution

Integrated WAN's native `text2video.WanT2V` module which provides:
1. Pre-trained VAE (Wan2_1_VAE) with proper latent encoding
2. Pre-trained T5 text encoder (T5EncoderModel) with built-in tokenization
3. Built-in distribution strategies (FSDP, sequence parallel, CPU offloading)
4. Proper shape handling for 1-frame video generation

## Key Changes

### 1. Imports (train_distillation.py)

**Before:**
```python
from diffusers import DiffusionPipeline, WanPipeline, AutoencoderKLWan
```

**After:**
```python
from easydict import EasyDict
from wan.text2video import WanT2V
from wan.configs.wan_t2v_A14B import t2v_A14B
```

### 2. Teacher Model Loading

**Before:**
- Used `WanPipeline.from_pretrained()` from diffusers
- Manual device mapping and dtype configuration
- Complex error handling for local vs. remote loading

**After:**
- Uses `WanT2V()` constructor with native WAN config
- Built-in support for t5_cpu, offload_model, FSDP, sequence parallel
- Simpler configuration through EasyDict config

```python
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
    convert_model_dtype=False
)
```

### 3. Text Encoding

**Before:**
- Manual tokenizer calls
- Complex device/dtype handling
- Fixed max_length truncation

**After:**
- WAN T5EncoderModel handles tokenization internally
- Returns list of variable-length embeddings
- Automatic padding and batching

```python
text_embeddings_list = teacher_text_encoder(prompts, device)
# Returns list of [seq_len, 4096] tensors, one per prompt
```

### 4. Latent Shape and Projection

**Before:**
- Used teacher transformer's `in_channels` (ambiguous)
- Manual shape conversions

**After:**
- Uses VAE `z_dim` (16 channels) - proper latent space
- Batch projection operation for efficiency
- Direct shape extraction without unnecessary copies

```python
vae_z_dim = teacher_vae.model.z_dim  # 16 for WAN VAE
# Projection: student_channels -> vae_z_dim
```

### 5. Teacher Forward Pass

**Before:**
```python
teacher_output = teacher_model(
    hidden_states=teacher_latents,
    timestep=timesteps_teacher,
    encoder_hidden_states=text_embeddings_teacher,
)
```

**After:**
```python
teacher_output_list = teacher_model(
    x=teacher_latents_list,  # List of [C, F, H, W]
    t=timesteps_teacher,     # [B]
    context=text_embeddings_teacher,  # List of [L, C]
    seq_len=seq_len,         # int
)
```

### 6. Distribution Strategies

**Before:**
- Manual device_map configuration
- Complex balanced/gpu0/cpu logic
- No FSDP or sequence parallel support

**After:**
- Built-in FSDP support via `dit_fsdp=True, t5_fsdp=True`
- Built-in sequence parallel via `use_sp=True`
- Automatic model offloading via `offload_model=True`
- CPU offloading for T5 via `t5_cpu=True`

## Memory Optimization Benefits

1. **FSDP (Fully Sharded Data Parallel)**
   - Shards model parameters across GPUs
   - ~8x memory reduction with 8 GPUs
   - Enabled via `dit_fsdp=True, t5_fsdp=True`

2. **Sequence Parallel**
   - Splits sequence dimension across GPUs
   - Reduces activation memory
   - Enabled via `use_sp=True`

3. **CPU Offloading**
   - Places T5 encoder on CPU
   - Saves ~20GB GPU memory
   - Enabled via `t5_cpu=True`

4. **Model Offloading**
   - Offloads inactive DiT models to CPU
   - Uses low_noise_model or high_noise_model based on timestep boundary
   - Enabled via `offload_model=True` in generate()

## Shape Specifications

### Student Model (2D Image)
- Latent: `[B, num_channels, H, W]` (e.g., `[2, 4, 16, 16]`)
- Output: `[B, num_channels, H, W]`

### Teacher Model (3D Video, 1-frame)
- Latent: List of `[C, 1, H, W]` tensors (1 per batch element)
- After projection: List of `[vae_z_dim, 1, H, W]` = `[16, 1, H, W]`
- Text: List of `[L, 4096]` tensors (variable length, padded)
- Output: List of `[C, 1, H, W]` → stacked to `[B, C, 1, H, W]` → squeezed to `[B, C, H, W]`

### VAE Specifications
- Latent channels (z_dim): 16
- Stride: (4, 8, 8) for temporal, height, width
- For 1-frame: temporal stride is effectively ignored

## Dependencies Added

```
easydict>=1.9   # Required for WAN configs
einops>=0.6.0   # Required for WAN VAE
```

## Testing

Run the validation script:
```bash
python test_wan_integration.py
```

This tests:
- WAN module imports
- Config structure
- train_distillation imports
- Shape compatibility
- Text encoder interface

## Backward Compatibility

The changes maintain backward compatibility with existing CLI arguments:
- `--teacher_device_strategy` still works (maps to WAN config)
- `--teacher_dtype` still works (maps to WAN config.param_dtype)
- `--teacher_on_cpu` still works (maps to t5_cpu=True)
- All existing memory optimization flags are preserved

## Performance Expectations

With WAN integration:
- **Memory usage**: ~30-50% reduction with proper distribution strategies
- **CUDA OOM**: Significantly reduced through FSDP and offloading
- **Training speed**: Comparable or slightly faster due to optimized implementations
- **1-frame generation**: Properly aligned with WAN's video model for image generation

## Migration Notes

If upgrading from the old implementation:
1. Install new dependencies: `pip install -r requirements.txt`
2. Ensure checkpoint directory has WAN format:
   - `models_t5_umt5-xxl-enc-bf16.pth`
   - `Wan2.1_VAE.pth`
   - `low_noise_model/`
   - `high_noise_model/`
3. Use `--teacher_device_strategy balanced` for multi-GPU OOM mitigation
4. Use `--teacher_device_strategy cpu` for extreme memory constraints
5. Test with small batch sizes first

## Future Work

Potential improvements:
1. Support for multi-frame training (currently 1-frame only)
2. Integration of WAN's generate() method for full inference pipeline
3. Support for image-to-video mode using y parameter
4. More aggressive memory optimizations using gradient checkpointing on teacher

## References

- WAN text2video module: `/wan/text2video.py`
- WAN config: `/wan/configs/wan_t2v_A14B.py`
- WAN VAE: `/wan/modules/vae2_1.py`
- WAN T5: `/wan/modules/t5.py`
- WAN distribution: `/wan/distributed/`
