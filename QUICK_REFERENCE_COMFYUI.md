# Quick Reference: ComfyUI Model Loading

## Question
> "Can you confirm how we need to load in comfy, will it be loaded in diffusion_models or diffusers or checkpoint?"

## Answer

### ✅ CORRECT Location
```
ComfyUI/models/checkpoints/
```

### ❌ WRONG Locations
- `ComfyUI/models/diffusion_models/` - Not used for this model type
- `ComfyUI/models/diffusers/` - Not a standard ComfyUI folder

## Why 'checkpoints' Folder?

WanLiteStudent models are saved in **HuggingFace Diffusers format** as complete model packages:
- `config.json` - Model configuration with metadata
- `diffusion_model.safetensors` - Model weights

ComfyUI loads these complete model packages from the `checkpoints/` folder.

## Quick Command

```bash
# Copy your trained model to ComfyUI
cp -r output/checkpoints/epoch_10 /path/to/ComfyUI/models/checkpoints/my_model
```

## Required File Structure

```
ComfyUI/models/checkpoints/my_model/
├── config.json                     # MUST contain "_class_name": "WanLiteStudent"
└── diffusion_model.safetensors     # Model weights
```

## Full Documentation

See [COMFYUI_LOADING_GUIDE.md](COMFYUI_LOADING_GUIDE.md) for:
- Complete step-by-step instructions
- Troubleshooting common errors
- File format requirements
- Model compatibility details

## Key Points

1. **Always use `checkpoints/` folder** - Not diffusion_models or diffusers
2. **Copy the entire directory** - Both config.json and .safetensors files
3. **Ensure `_class_name` is present** - Required for ComfyUI to load the model
4. **Restart ComfyUI if needed** - To refresh the model list

## Related Files

- [COMFYUI_LOADING_GUIDE.md](COMFYUI_LOADING_GUIDE.md) - Comprehensive loading guide
- [COMFY_UI_FIX_SUMMARY.md](COMFY_UI_FIX_SUMMARY.md) - Technical details about the ComfyUI fix
- [docs/WAN_COMPATIBLE_SAVE_FORMAT.md](docs/WAN_COMPATIBLE_SAVE_FORMAT.md) - Save format specification
