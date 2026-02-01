# Quick Reference: UMT5 Weight Loading Warning

## The Warning You See

```
UMT5EncoderModel LOAD REPORT from: ...text_encoder
Key                         | Status  | Details
----------------------------+---------+--------
encoder.embed_tokens.weight | MISSING |
```

## What It Means

✅ **This is NORMAL and EXPECTED**  
✅ **Your model is loading CORRECTLY**  
✅ **No action needed from you**

## Why It Happens

The UMT5 text encoder uses "weight tying":
- `shared.weight` contains the actual embeddings (loaded ✓)
- `encoder.embed_tokens.weight` references `shared.weight` (not stored separately)
- The warning appears because the checkpoint doesn't have a separate copy
- But the model automatically creates the reference after loading

## Quick Fix

If you want to suppress the warning, the latest version of `train_distillation.py` already handles it automatically.

## Need More Details?

See: [docs/UMT5_WEIGHT_LOADING.md](UMT5_WEIGHT_LOADING.md)

---

**Bottom Line:** Ignore the warning and proceed with training. Your model is working correctly! 🎉
