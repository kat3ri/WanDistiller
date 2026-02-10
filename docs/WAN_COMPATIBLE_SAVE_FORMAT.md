# WAN-Compatible Save Format

## Overview

The WanLiteStudent model now uses HuggingFace Diffusers' `ModelMixin` and `ConfigMixin` to save models in a format that is fully compatible with the WAN model structure and can be loaded into ComfyUI.

## Key Changes

### 1. Class Structure

**Before:**
```python
class WanLiteStudent(nn.Module):
    def __init__(self, config, teacher_checkpoint_path=None, ...):
        super().__init__()
        self.config = config
        # ... model initialization
```

**After:**
```python
class WanLiteStudent(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        model_type="WanLiteStudent",
        hidden_size=1024,
        depth=16,
        num_heads=16,
        # ... all config parameters explicitly defined
    ):
        super().__init__()
        # ... model initialization
```

### 2. Benefits

1. **Automatic Config Management**: The `@register_to_config` decorator automatically captures all initialization parameters and saves them to `config.json`

2. **HuggingFace Compatibility**: Models can now be loaded using the standard `from_pretrained()` method:
   ```python
   model = WanLiteStudent.from_pretrained("path/to/saved/model")
   ```

3. **WAN Model Structure**: Saved models follow the same structure as WAN models:
   ```
   saved_model/
   ├── config.json                    # Model configuration
   └── diffusion_model.safetensors    # Model weights
   ```

4. **ComfyUI Compatible**: Models saved in this format can be loaded directly into ComfyUI as if they were WAN models.

### 3. Backward Compatibility

The model still supports the old calling convention for backward compatibility with existing code:

```python
# Old style (still works)
config = {
    'model_type': 'WanLiteStudent',
    'hidden_size': 1024,
    # ... other parameters
}
model = WanLiteStudent(config, teacher_checkpoint_path=None, device='cuda:0')

# New style (recommended)
model = WanLiteStudent(
    model_type='WanLiteStudent',
    hidden_size=1024,
    depth=16,
    # ... other parameters
    teacher_checkpoint_path=None,
    device='cuda:0'
)
```

## Saved Configuration

When you call `model.save_pretrained(output_dir)`, the following is saved:

### config.json
Contains all model architecture parameters:
- `_class_name`: Model class name (default: "WanLiteStudent") - **Required by ComfyUI**
- `_diffusers_version`: Diffusers library version for compatibility tracking
- `model_type`: Model identifier (default: "WanLiteStudent")
- `hidden_size`: Hidden dimension
- `depth`: Number of transformer blocks
- `num_heads`: Number of attention heads
- `num_channels`: Number of input/output channels
- `image_size`: Input image size
- `patch_size`: Patch size for image tokenization
- `text_max_length`: Maximum text sequence length
- `text_encoder_output_dim`: Text encoder output dimension
- `projection_factor`: Factor for weight projection from teacher

### diffusion_model.safetensors
Contains all model weights in the efficient safetensors format.

## Usage Examples

### Saving a Model

```python
import train_distillation

# Initialize model
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

# Train model...
# ...

# Save in WAN-compatible format
student_model.save_pretrained("./output/student_model")
```

Output:
```
✓ Model saved successfully to: ./output/student_model
  - config.json (model configuration)
  - diffusion_model.safetensors (model weights)
  Format: HuggingFace Diffusers (compatible with WAN and ComfyUI)
```

### Loading a Model

```python
import train_distillation

# Load model using standard HuggingFace method
model = train_distillation.WanLiteStudent.from_pretrained("./output/student_model")

# Model is ready to use
model.eval()
```

### Loading in ComfyUI

The saved model can be placed in ComfyUI's models directory and loaded using the standard WAN model loading nodes:

1. Copy the saved model directory to ComfyUI's models folder:
   ```
   ComfyUI/models/checkpoints/student_model/
   ├── config.json
   └── diffusion_model.safetensors
   ```

2. In ComfyUI, use the model loader node with the path to the model directory

3. The model will be loaded with all the correct metadata and configuration

## Migration Guide

If you have existing code that uses the old save format:

### Old Code
```python
# Old way - saved to student_config.json
model.save_pretrained(output_dir)

# Files saved:
# - output_dir/student_config.json
# - output_dir/diffusion_model.safetensors
```

### New Code
```python
# New way - saves to config.json (WAN-compatible)
model.save_pretrained(output_dir)

# Files saved:
# - output_dir/config.json
# - output_dir/diffusion_model.safetensors
```

**Note**: The new format uses `config.json` instead of `student_config.json` to match the WAN model structure.

## Technical Details

### ModelMixin Features

The `ModelMixin` base class provides:
- `save_pretrained()`: Saves model weights in safetensors format
- `from_pretrained()`: Loads model from a directory
- Device management utilities
- Gradient checkpointing support

### ConfigMixin Features

The `ConfigMixin` base class provides:
- Automatic config serialization to JSON
- Config loading and validation
- Version tracking for models

### @register_to_config Decorator

This decorator automatically:
1. Captures all __init__ parameters
2. Stores them in `self.config`
3. Makes them available for serialization
4. Validates parameter types

## Testing

Run the structure test to verify the changes:
```bash
python test_model_structure.py
```

This will verify:
- Proper inheritance from ModelMixin and ConfigMixin
- @register_to_config decorator is correctly applied
- All required config parameters are present
- save_pretrained uses the parent class method

## Troubleshooting

### Issue: Model doesn't load in ComfyUI

**Solution**: Ensure the model directory contains:
1. `config.json` (not `student_config.json`)
2. `diffusion_model.safetensors` (not other weight formats)
3. `_class_name` field in config.json (automatically added as of this fix)

If you have an old model without `_class_name`, re-save it with the updated code to add the required metadata.

### Issue: 'NoneType' object has no attribute 'clone' error in ComfyUI

**Solution**: This error occurs when the `_class_name` field is missing from `config.json`. The fix automatically adds this field when saving models. Re-save your model using `model.save_pretrained()` to include the required metadata.

See `COMFY_UI_FIX_SUMMARY.md` for detailed information about this fix.

### Issue: Config parameters missing

**Solution**: Make sure all parameters passed to `__init__` are listed before the `teacher_checkpoint_path` parameter, as only those are registered to the config.

### Issue: Backward compatibility issues

**Solution**: The model supports both dict-based config (old style) and individual parameters (new style). If you encounter issues, check that your config dict contains all required keys.

## References

- [HuggingFace Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [ModelMixin API Reference](https://huggingface.co/docs/diffusers/api/models/overview)
- [WAN Model Documentation](wan/README.md)
