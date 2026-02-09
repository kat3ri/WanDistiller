# Fix Summary: Model Loading and Inference Issues

## Problem Statement

When running inference with a saved WanLiteStudent model, two critical issues occurred:

1. **Unwanted Projection at Inference**: During model loading, the teacher weight projection process was being triggered again, which is unnecessary and time-consuming during inference.

2. **Meta Tensor Error**: The model loading failed with the error:
   ```
   Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() 
   instead of torch.nn.Module.to() when moving module from meta to a different device.
   ```

## Root Causes

### Issue 1: Projection Triggering
The `WanLiteStudent.__init__()` method was designed to optionally load teacher weights and perform projection when `teacher_checkpoint_path` is provided. However, when using the inherited `ModelMixin.from_pretrained()` method, the initialization process could inadvertently trigger this projection logic even during inference loading.

### Issue 2: Meta Tensor Handling
The default `ModelMixin.from_pretrained()` method may initialize models using meta tensors (tensors without allocated memory) for efficiency. This approach works well for standard diffusers models but caused issues with our custom initialization logic.

## Solution

### Custom `from_pretrained()` Method
We implemented a custom `from_pretrained()` classmethod in `WanLiteStudent` that:

1. **Loads Configuration First**: Reads `config.json` to get model architecture parameters

2. **Initializes Without Projection**: Explicitly sets `teacher_checkpoint_path=None` when creating the model instance, ensuring no teacher weight projection occurs

3. **Loads Weights Directly**: Uses `safetensors.torch.load_file()` to load weights directly on CPU, avoiding meta tensor issues

4. **Supports Multiple Filename Conventions**: Handles both `diffusion_pytorch_model.safetensors` (standard diffusers format) and `diffusion_model.safetensors` (custom format)

5. **Proper Device Handling**: Loads weights on CPU first, then moves to the target device as specified

### Code Changes

#### train_distillation.py
Added custom `from_pretrained()` method to `WanLiteStudent` class:

```python
@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    """Load a pretrained WanLiteStudent model from a directory."""
    from safetensors.torch import load_file
    import json
    
    # 1. Load config
    config_path = os.path.join(pretrained_model_name_or_path, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 2. Initialize WITHOUT teacher_checkpoint_path (skip projection!)
    model = cls(
        model_type=config_dict.get('model_type', 'WanLiteStudent'),
        hidden_size=config_dict['hidden_size'],
        depth=config_dict['depth'],
        num_heads=config_dict['num_heads'],
        num_channels=config_dict['num_channels'],
        image_size=config_dict['image_size'],
        patch_size=config_dict['patch_size'],
        text_max_length=config_dict['text_max_length'],
        text_encoder_output_dim=config_dict['text_encoder_output_dim'],
        projection_factor=config_dict.get('projection_factor', 1.0),
        teacher_checkpoint_path=None,  # CRITICAL: Don't trigger projection!
        distributed=False,
        use_gradient_checkpointing=False
    )
    
    # 3. Load weights (supports multiple filenames)
    possible_names = [
        "diffusion_pytorch_model.safetensors",
        "diffusion_model.safetensors",
        "model.safetensors"
    ]
    
    weights_path = None
    for name in possible_names:
        candidate_path = os.path.join(pretrained_model_name_or_path, name)
        if os.path.exists(candidate_path):
            weights_path = candidate_path
            break
    
    # Load on CPU first to avoid device issues
    state_dict = load_file(weights_path, device="cpu")
    model.load_state_dict(state_dict)
    
    # 4. Move to target device if specified
    device = kwargs.get('device', None)
    if device is not None:
        model = model.to(device)
    
    return model
```

#### run_inference.py
Updated error message to reflect supported filename formats:

```python
print(f"   - config.json (model configuration)", file=sys.stderr)
print(f"   - diffusion_pytorch_model.safetensors or diffusion_model.safetensors (model weights)", file=sys.stderr)
```

## Benefits

1. **✅ No Projection During Inference**: Loading a saved model no longer triggers teacher weight projection, making inference startup much faster

2. **✅ No Meta Tensor Errors**: Direct weight loading avoids the meta tensor initialization path that caused errors

3. **✅ Backward Compatibility**: The code supports multiple safetensors filename conventions, ensuring compatibility with different save formats

4. **✅ Cleaner Separation**: Clear distinction between training (with projection) and inference (without projection) workflows

5. **✅ Proper Error Messages**: Users get clear, actionable error messages if model files are missing

## Testing

Created comprehensive integration test (`test_save_load_integration.py`) that:
- ✅ Simulates full training→save→load→inference workflow
- ✅ Verifies no projection occurs during loading
- ✅ Confirms weights and configuration integrity
- ✅ Tests for meta tensor error prevention

Test results: **All tests pass** ✅

## Usage

### During Training
```python
# Create model with teacher projection
student_model = WanLiteStudent(
    hidden_size=1024,
    depth=16,
    num_heads=16,
    teacher_checkpoint_path="./Wan2.2-T2V-A14B",  # Triggers projection
    ...
)

# Train the model...

# Save the trained model
student_model.save_pretrained("./outputs/wan_t2i")
```

### During Inference
```python
# Load the saved model (NO projection occurs!)
student_model = WanLiteStudent.from_pretrained("./outputs/wan_t2i")
student_model.to(device)
student_model.eval()

# Use for inference...
```

## Conclusion

The custom `from_pretrained()` method successfully resolves both issues:
1. Prevents unnecessary teacher weight projection during inference
2. Avoids meta tensor errors through direct weight loading

This fix ensures smooth, efficient model loading for inference while maintaining full compatibility with the training workflow.
