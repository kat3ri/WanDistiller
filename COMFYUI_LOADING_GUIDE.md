# ComfyUI Loading Guide for WanLiteStudent Models

## Quick Answer

**Where to place your model:** `ComfyUI/models/checkpoints/`

**NOT** in `diffusion_models/` or `diffusers/` folders.

## Detailed Instructions

### Step 1: Save Your Model with Proper Format

Ensure you're using the updated `save_pretrained()` method that includes ComfyUI metadata:

```python
import train_distillation

# Train or load your model
student_model = train_distillation.WanLiteStudent(
    model_type='WanLiteStudent',
    hidden_size=1024,
    depth=16,
    num_heads=16,
    num_channels=4,
    image_size=1024,
    patch_size=16,
    text_max_length=77,
    text_encoder_output_dim=4096,
    device='cuda:0'
)

# ... training code ...

# Save with ComfyUI-compatible format
student_model.save_pretrained("./output/my_student_model")
```

This will create:
```
output/my_student_model/
├── config.json                    # Contains _class_name and model config
└── diffusion_model.safetensors    # Model weights
```

### Step 2: Copy Model to ComfyUI

Copy the **entire model directory** to ComfyUI's checkpoints folder:

```bash
# Example: Copy the model directory
cp -r ./output/my_student_model/ /path/to/ComfyUI/models/checkpoints/
```

**Result:**
```
ComfyUI/
└── models/
    └── checkpoints/
        └── my_student_model/
            ├── config.json
            └── diffusion_model.safetensors
```

### Step 3: Load in ComfyUI

1. **Open ComfyUI**
2. **Add a model loader node** (specific to your workflow)
3. **Select your model** from the dropdown - look for `my_student_model`
4. **Connect to your workflow** and run

The model will be loaded with all metadata and should work without the `'NoneType' object has no attribute 'clone'` error.

## Important: Model Format Requirements

### ✅ Required Files
Your model directory **MUST** contain:

1. **config.json** with these critical fields:
   ```json
   {
     "_class_name": "WanLiteStudent",
     "_diffusers_version": "0.36.0",
     "model_type": "WanLiteStudent",
     "hidden_size": 1024,
     "depth": 16,
     "num_heads": 16,
     ...
   }
   ```

2. **diffusion_model.safetensors** - Model weights in safetensors format

### ❌ Common Mistakes

**Wrong folder locations:**
- ❌ `ComfyUI/models/diffusion_models/` - Not used for this model type
- ❌ `ComfyUI/models/diffusers/` - Not a standard ComfyUI folder
- ❌ `ComfyUI/models/unet/` - Wrong model type
- ✅ `ComfyUI/models/checkpoints/` - **CORRECT**

**Wrong file structure:**
- ❌ Copying just the `.safetensors` file without `config.json`
- ❌ Using old models without `_class_name` in config.json
- ✅ Copying the entire directory with both files

## Troubleshooting

### Error: 'NoneType' object has no attribute 'clone'

**Cause:** The `config.json` is missing the `_class_name` field.

**Solution:** 
1. Re-save your model using the updated `save_pretrained()` method
2. Verify `config.json` contains `"_class_name": "WanLiteStudent"`
3. Copy the updated model to ComfyUI

### Error: Model not appearing in ComfyUI dropdown

**Possible causes:**
1. Model is in the wrong folder
   - **Fix:** Move to `ComfyUI/models/checkpoints/`
   
2. Missing required files
   - **Fix:** Ensure both `config.json` and `diffusion_model.safetensors` are present
   
3. ComfyUI needs restart
   - **Fix:** Restart ComfyUI to refresh the model list

### Error: Model loads but doesn't work

**Possible causes:**
1. Config.json missing fields
   - **Fix:** Verify `_class_name` and all model parameters are present
   
2. Incompatible model architecture
   - **Fix:** Check logs for specific errors about layer mismatches

## Why 'checkpoints' Folder?

ComfyUI organizes models into different folders based on their type and format:

- **`checkpoints/`** - Complete model packages (used for full diffusion models with config + weights)
- **`unet/`** - Standalone UNet components
- **`vae/`** - VAE models
- **`clip/`** - Text encoder models
- **`loras/`** - LoRA adapters

Since WanLiteStudent is a complete diffusion model saved in HuggingFace Diffusers format (config.json + diffusion_model.safetensors), it belongs in the **checkpoints** folder.

## Model Format Compatibility

WanLiteStudent models use the **HuggingFace Diffusers** format:
- Inherits from `ModelMixin` and `ConfigMixin`
- Uses `@register_to_config` decorator
- Saved with `save_pretrained()` method
- Compatible with `from_pretrained()` loading

This format is:
- ✅ Compatible with ComfyUI (when placed in `checkpoints/`)
- ✅ Compatible with HuggingFace ecosystem
- ✅ Compatible with WAN model loading patterns
- ✅ Easy to share and distribute

## Example: Complete Workflow

```bash
# 1. Train your model
python train_distillation.py --config config/student_config.json

# 2. Model is automatically saved to output directory
# output/checkpoints/epoch_10/
# ├── config.json
# └── diffusion_model.safetensors

# 3. Copy to ComfyUI
cp -r output/checkpoints/epoch_10 /path/to/ComfyUI/models/checkpoints/my_model

# 4. Launch ComfyUI
cd /path/to/ComfyUI
python main.py

# 5. Use model in your workflow
# - Add model loader node
# - Select "my_model" from dropdown
# - Connect to your generation pipeline
```

## Verification Checklist

Before loading in ComfyUI, verify:

- [ ] Model saved using updated `save_pretrained()` method
- [ ] `config.json` exists and contains `_class_name` field
- [ ] `diffusion_model.safetensors` exists
- [ ] Both files are in the same directory
- [ ] Directory copied to `ComfyUI/models/checkpoints/`
- [ ] ComfyUI restarted (if needed)

## Related Documentation

- `COMFY_UI_FIX_SUMMARY.md` - Details about the ComfyUI compatibility fix
- `docs/WAN_COMPATIBLE_SAVE_FORMAT.md` - Model save format specification
- `SAVE_LOGIC_REVIEW_SUMMARY.md` - Technical details about save logic

## Support

If you continue to have issues:
1. Check that you're using the latest version of the code
2. Verify `_class_name` is in your config.json
3. Check ComfyUI logs for specific error messages
4. Ensure your ComfyUI version supports HuggingFace Diffusers format models
