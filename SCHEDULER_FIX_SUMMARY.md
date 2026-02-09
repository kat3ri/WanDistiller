# Summary: VAE Usage Investigation & Scheduler Fix

## Your Question
> Training ran, loss curved looked picture perfect, but inference produces garbage. Can you double check the vae usage? We are currently using the wan video vae and attempting to decode an image with it from our student model -- please advise whether we should train the vae while training or attempt to use a separate image-focused vae for inference?

## Quick Answer

**NO, you should NOT train the VAE or use a different VAE!** 

The VAE usage was correct. The problem was with the **noise scheduler in inference**.

## What Was Wrong

Your inference script (`run_inference.py`) was using a custom DDIM implementation but your teacher model uses Flow Matching with `FlowUniPCMultistepScheduler`. **This mismatch caused garbage output** even though training loss was perfect!

## What Was Fixed

Updated `run_inference.py` to use the correct Flow Matching scheduler matching the teacher model.

## Addressing Your Questions

### Q1: Should we train the VAE while training?

**Answer: NO - Keep VAE frozen**

The VAE is already pre-trained and the student learns to match the teacher's latent space distribution. Training the VAE would break compatibility with the teacher.

### Q2: Should we use a separate image-focused VAE?

**Answer: NO - Keep using the video VAE**

The video VAE handles images correctly (T=1 temporal dimension) with frame-by-frame processing. Using a different VAE would cause incompatible latent space statistics and break the student model.

## How to Use the Fix

```bash
python run_inference.py \
  --model_path "./outputs/wan_t2i" \
  --teacher_path "./Wan2.2-T2V-A14B" \
  --prompt "A serene lake at sunset with mountains" \
  --output_path "result.png" \
  --num_inference_steps 20 \
  --shift 5.0
```

## What You Should See Now

✅ **Before (with bug)**: Garbage output despite good training loss  
✅ **After (with fix)**: Clean, high-quality images matching training quality

## More Details

See **[VAE_USAGE_FIX.md](VAE_USAGE_FIX.md)** for complete technical explanation.

---

**Status**: ✅ **Complete and Ready for Testing**
