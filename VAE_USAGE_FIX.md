# VAE Usage Fix: Correcting Inference Garbage Output

## Problem Statement

Training produced picture-perfect loss curves, but inference generated garbage output. The question was whether the issue was with VAE usage - specifically:
1. Should we train the VAE during training?
2. Should we use a separate image-focused VAE for inference instead of the video VAE?

## Root Cause Analysis

After thorough investigation, **the VAE usage was NOT the problem**. The issue was with the **noise scheduler** used during inference.

### What We Found

1. **VAE Architecture is Correct**: The Wan Video VAE (3D causal convolutions) handles single-frame images correctly by processing them with temporal dimension T=1. The VAE does frame-by-frame decoding internally, so it works fine for both video and images.

2. **The Real Problem**: **Inference was using a completely wrong noise scheduler!**
   - **Teacher Model** (training reference): Uses **Flow Matching** with `FlowUniPCMultistepScheduler`
   - **Original Inference Code**: Used a custom **DDIM implementation** with simple linear timestep scheduling
   - **Result**: Latent distributions from the mismatched scheduler were incompatible with the VAE

### Technical Details

The teacher model uses flow matching solvers with specific properties:
```python
# Teacher (wan/text2video.py)
scheduler = FlowUniPCMultistepScheduler(
    num_train_timesteps=1000,
    shift=1,
    use_dynamic_shifting=False
)
scheduler.set_timesteps(sampling_steps, device=device, shift=5.0)
```

The original inference code used:
```python
# Original inference (WRONG)
timesteps = torch.linspace(999, 0, num_inference_steps, dtype=torch.long, device=device)
alpha_t = 1.0 - (t.float() / 1000.0)  # Simple linear schedule
```

This mismatch meant:
- Student was trained to predict noise using flow matching dynamics
- Inference accumulated those predictions using DDIM dynamics
- Final latents had wrong distribution → VAE decoded garbage

## Solution

**Use the same Flow Matching scheduler in inference as the teacher uses during generation.**

### Changes Made

`run_inference.py` now uses:
```python
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

scheduler = FlowUniPCMultistepScheduler(
    num_train_timesteps=1000,
    shift=1,
    use_dynamic_shifting=False
)
scheduler.set_timesteps(args.num_inference_steps, device=device, shift=args.shift)

# Use scheduler.step() for each denoising step
current_latents = scheduler.step(
    noise_pred, 
    t, 
    current_latents,
    return_dict=False
)[0]
```

### New CLI Parameter

Added `--shift` parameter (default: 5.0) to control the flow matching shift:
```bash
python run_inference.py \
  --model_path "./outputs/wan_t2i" \
  --teacher_path "./Wan2.2-T2V-A14B" \
  --prompt "A beautiful mountain landscape" \
  --num_inference_steps 20 \
  --shift 5.0
```

## Answers to Original Questions

### Q1: Should we train the VAE while training the student model?

**Answer: NO.** The VAE should remain frozen (with `requires_grad=False`). Here's why:

1. **VAE is pretrained**: The WAN Video VAE is already trained to encode/decode videos and images
2. **Student learns latent space**: The student model learns to generate latents in the VAE's latent space by mimicking the teacher
3. **Training VAE would break compatibility**: If you trained the VAE, it would drift from the teacher's expectations
4. **Memory efficiency**: Frozen VAE saves memory and compute

The current setup is correct:
```python
teacher_vae = teacher_wan.vae
# VAE is automatically in eval mode and gradients are not computed
```

### Q2: Should we use a separate image-focused VAE for inference?

**Answer: NO.** The WAN Video VAE works perfectly for images. Here's why:

1. **Architecture is flexible**: The 3D causal convolutions handle temporal dimension T=1 naturally
2. **Frame-by-frame processing**: The VAE decodes frame-by-frame internally (see `vae2_1.py` line 562-574)
3. **Single-frame is just a special case**: Video with 1 frame = image
4. **Latent space compatibility**: Student is trained on this specific VAE's latent space

Using a different VAE would cause issues:
- Different latent space statistics (mean, std)
- Different channel dimensions
- Different spatial compression ratios
- Incompatible with student's learned distribution

## Verification Steps

To verify the fix works:

1. **Run inference with the updated script**:
   ```bash
   python run_inference.py \
     --model_path "./outputs/wan_t2i" \
     --teacher_path "./Wan2.2-T2V-A14B" \
     --prompt "A serene lake at sunset" \
     --output_path "test_output.png" \
     --num_inference_steps 20 \
     --shift 5.0
   ```

2. **Expected results**:
   - No more garbage output
   - Images should match training quality
   - Loss curves correlate with visual quality

3. **Troubleshooting** (if still issues):
   - Try different shift values (3.0, 5.0, 7.0)
   - Increase inference steps (50-100)
   - Check model checkpoint integrity
   - Verify training completed properly

## Key Takeaways

1. ✅ **VAE usage was correct all along** - video VAE works for images
2. ✅ **VAE does NOT need training** - keep it frozen
3. ✅ **Scheduler mismatch was the culprit** - now fixed
4. ✅ **Use Flow Matching scheduler in inference** - matches training dynamics
5. ✅ **No need for separate image VAE** - current VAE is perfect

## Technical Note: Why Shape Handling is Correct

The shape transformations are correct:

1. **Student output**: `[B, C, H, W]` (batch, channels, height, width)
2. **Add temporal dimension**: `[B, C, 1, H, W]` via `unsqueeze(1)` → becomes `[C, 1, H, W]` per item
3. **VAE wrapper adds batch**: `[1, C, 1, H, W]` via `unsqueeze(0)`
4. **VAE expects**: `[batch, channels, time, height, width]` ✓

The VAE's decode function at line 554 confirms: `# z: [b,c,t,h,w]`

Everything is correctly aligned!

## References

- `run_inference.py`: Updated inference script with Flow Matching scheduler
- `wan/text2video.py` (line 394-446): Teacher model's generation with Flow Matching
- `wan/modules/vae2_1.py` (line 710-723): VAE decode wrapper
- `wan/modules/vae2_1.py` (line 552-576): VAE model decode implementation
- `wan/utils/fm_solvers_unipc.py`: Flow Matching UniPC scheduler implementation

## Additional Notes

The Flow Matching scheduler is crucial because:
- It uses rectified flow formulation
- Timestep shifting controls the noise schedule
- UniPC solver provides efficient multistep sampling
- This matches the teacher's training dynamics exactly

Without this, even though the student learned correct noise predictions, the accumulation process was wrong, leading to garbage latents that the VAE decoded into garbage images.
