# Inference Fix: DDIM Sampling Implementation

## Problem Statement
After training with perfect loss metrics, inference was producing only noise instead of coherent images. This issue was critical as it prevented the trained model from being used for its intended purpose.

## Root Cause Analysis

### The Issue
The training and inference pipelines were using **fundamentally different sampling methods**:

- **Training sample generation** (train_distillation.py, lines 799-856): Used custom DDIM (Denoising Diffusion Implicit Models) sampling
- **Inference** (run_inference.py, original): Used FlowUniPCMultistepScheduler (Flow Matching)

### Why This Caused Noise
Flow Matching and DDIM are completely different approaches to the diffusion process:

1. **DDIM**: Uses a linear noise schedule with alpha-based denoising steps
   - Timesteps: 999 → 0 (high noise to clean)
   - Alpha schedule: α_t = 1 - t/1000
   - Deterministic sampling process

2. **Flow Matching**: Uses a different parameterization based on continuous normalizing flows
   - Different timestep interpretation
   - Different velocity field prediction
   - Requires shift parameters for control

Since the model was trained to work with DDIM's noise schedule and prediction targets, using Flow Matching at inference time caused:
- Incorrect timestep interpretations
- Mismatched prediction targets
- Improper denoising steps
- Result: Pure noise output

## Solution

Replaced Flow Matching with the **exact DDIM sampling implementation from training**:

### Changes Made

1. **Removed Flow Matching Dependencies**
   ```python
   # Removed:
   from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
   ```

2. **Implemented DDIM Sampling**
   ```python
   # Create timestep schedule from high (999) to low (0)
   timesteps = torch.linspace(999, 0, args.num_inference_steps, dtype=torch.long, device=device)
   
   # DDIM sampling loop with alpha-based denoising
   for i, t in enumerate(timesteps):
       alpha_t = 1.0 - (t.float() / 1000.0)
       # ... (complete DDIM update step)
   ```

3. **Removed Flow Matching Parameter**
   - Removed `--shift` argument (only applicable to Flow Matching)

### Key Implementation Details

The DDIM implementation includes:

- **Linear timestep schedule**: `torch.linspace(999, 0, num_steps)` 
- **Alpha calculation**: `α_t = 1 - t/1000` where high t (999) → low α, low t (0) → high α
- **Numerical stability**: Epsilon clamping (1e-3) for alpha values
- **DDIM update formula**:
  ```
  x̂_0 = (x_t - √(1-α_t) * ε_θ) / √α_t
  x_{t-1} = √α_{t-1} * x̂_0 + √(1-α_{t-1}) * ε_θ
  ```

## Verification

### Before Fix
- Training: Perfect loss metrics
- Sample generation during training: Coherent images
- Inference: Only noise

### After Fix
- Inference now uses identical sampling method as training
- Expected behavior: Coherent images matching training sample quality

## Usage

```bash
python run_inference.py \
  --model_path "./outputs/wan_t2i" \
  --teacher_path "./Wan2.2-T2V-A14B" \
  --prompt "A beautiful mountain landscape at sunset" \
  --output_path "result.png" \
  --num_inference_steps 20
```

**Note**: The `--shift` parameter has been removed as it's no longer applicable with DDIM sampling.

## Technical Notes

### Why Exact Matching is Critical
The implementation **exactly matches** the training code (train_distillation.py lines 799-856), including:
- Same alpha calculation formula
- Same epsilon value (1e-3) for numerical stability  
- Same handling of the final timestep
- Same DDIM update equations

Any deviation, even seemingly minor optimizations, could introduce subtle differences that affect output quality.

### Numerical Stability Considerations
The code includes epsilon clamping to prevent division by zero or very small numbers:
```python
alpha_t = torch.clamp(alpha_t, min=epsilon, max=1.0 - epsilon)
```

While there are theoretical concerns about dividing by small sqrt(alpha_t) values, this implementation matches the training code exactly, which has proven to work effectively.

## References

- DDIM Paper: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- Training implementation: `train_distillation.py` lines 799-856
- Flow Matching: Different approach that requires different training

## Future Considerations

If Flow Matching is desired for inference:
1. The model would need to be **retrained** using Flow Matching
2. Both training and inference would need to use the same Flow Matching scheduler
3. The training loss and sampling code would need to be updated accordingly

The key principle: **Training and inference must use the same sampling method**.
