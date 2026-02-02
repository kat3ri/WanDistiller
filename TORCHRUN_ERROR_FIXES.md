# Torchrun ChildFailedError - Explanation and Fixes

## Error Explanation

When you see this error:

```
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
train_distillation.py FAILED
```

This is a **generic error from PyTorch's distributed launcher (torchrun)** indicating that one or more child processes failed during execution. The actual error message is often hidden because:

1. Torchrun spawns multiple child processes (one per GPU)
2. Error messages from child processes may be suppressed or not visible
3. The parent process only reports that children failed, not why

## Common Causes and Solutions

### 1. Incorrect Command Syntax ❌

**Problem:**
```bash
torchrun --nproc_per_node=4 python train_distillation.py ...
```

**Why it fails:** When you include `python` before the script name, torchrun thinks the script name is `python`, which triggers an early exit in the validation check.

**Solution:**
```bash
torchrun --nproc_per_node=4 train_distillation.py ...
```

**Explanation:** Torchrun already invokes Python internally, so you should NOT include `python` in the command.

---

### 2. CUDA Not Available ❌

**Problem:** Running with `--distributed` flag but CUDA is not available.

**Error message you'll now see:**
```
[Rank 0] ERROR: CUDA is not available!
Distributed training with NCCL backend requires CUDA-enabled GPUs.
```

**Solutions:**
1. Ensure CUDA is properly installed:
   ```bash
   nvidia-smi  # Check if GPUs are detected
   python -c "import torch; print(torch.cuda.is_available())"  # Check PyTorch CUDA
   ```

2. Run without distributed training:
   ```bash
   python train_distillation.py ... (remove torchrun)
   ```

---

### 3. Requesting More GPUs Than Available ❌

**Problem:**
```bash
torchrun --nproc_per_node=4 train_distillation.py ...  # But only 2 GPUs available
```

**Error message you'll now see:**
```
[Rank 2] ERROR: Invalid GPU configuration!
This process (rank 2, local_rank 2) is trying to use GPU index 2,
but only 2 GPU(s) are available on this machine (GPU indices 0 to 1).
```

**Solutions:**
1. Reduce `--nproc_per_node` to match available GPUs:
   ```bash
   nvidia-smi  # Check how many GPUs you have
   torchrun --nproc_per_node=2 train_distillation.py ...  # Use correct number
   ```

2. Run on CPU without distributed training:
   ```bash
   python train_distillation.py ...
   ```

---

### 4. NCCL Backend Initialization Failure ❌

**Problem:** PyTorch's NCCL backend fails to initialize the distributed process group.

**Error message you'll now see:**
```
[Rank 0] ERROR: Failed to initialize distributed process group
Error details: <specific error>

Common causes:
  1. NCCL backend not properly installed or configured
  2. Network issues preventing inter-process communication
  3. Mismatched PyTorch/CUDA versions
```

**Solutions:**
1. Verify NCCL is available:
   ```bash
   python -c 'import torch; print(torch.cuda.nccl.is_available())'
   ```

2. Check PyTorch installation:
   ```bash
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   ```

3. Reinstall PyTorch with proper CUDA support if needed:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

---

### 5. GPU Device Set Failure ❌

**Problem:** Unable to set the CUDA device for a specific rank.

**Error message you'll now see:**
```
[Rank 1] ERROR: Failed to set CUDA device 1
Error details: <specific error>

This usually means:
  - GPU 1 is not accessible
  - CUDA driver issue
  - GPU 1 is being used by another process
```

**Solutions:**
1. Check GPU status:
   ```bash
   nvidia-smi
   ```

2. Free up GPU memory:
   ```bash
   # Find processes using GPU
   nvidia-smi
   # Kill processes if needed
   kill <PID>
   ```

3. Use a different GPU or fewer processes

---

### 6. Teacher Model Loading Failure ❌

**Problem:** Required teacher model files are missing or corrupted.

**Error message you'll now see:**
```
ERROR: Failed to load WAN teacher model
Model path: <path>

Please ensure the checkpoint directory contains:
  1. models_t5_umt5-xxl-enc-bf16.pth (T5 encoder)
  2. Wan2.1_VAE.pth (VAE)
  3. low_noise_model/ (DiT low noise)
  4. high_noise_model/ (DiT high noise)
```

**Solutions:**
1. Download the correct teacher model:
   ```bash
   python download_wan.py
   ```

2. Verify the checkpoint path is correct:
   ```bash
   ls -la <path>  # Check if all required files exist
   ```

---

### 7. CUDA Out of Memory ❌

**Problem:** Model or batch size too large for available GPU memory.

**Error message you'll now see:**
```
[Rank 0] ERROR: CUDA Out of Memory!
GPU Memory:
  Allocated: X.XX GB
  Total: Y.YY GB

Memory-saving solutions (try in order):
  1. Load teacher on CPU: Add --teacher_on_cpu flag
  2. Use lower precision: Add --teacher_dtype float16
  3. Reduce batch size: --batch_size 1
  ...
```

**Solutions:**
Try these in order:

1. **Load teacher on CPU** (saves ~120GB GPU memory):
   ```bash
   torchrun --nproc_per_node=1 train_distillation.py \
     --teacher_on_cpu \
     ... other args ...
   ```

2. **Use lower precision** (saves ~50% memory):
   ```bash
   torchrun --nproc_per_node=1 train_distillation.py \
     --teacher_dtype float16 \
     ... other args ...
   ```

3. **Reduce batch size**:
   ```bash
   torchrun --nproc_per_node=1 train_distillation.py \
     --batch_size 1 \
     ... other args ...
   ```

4. **Enable gradient checkpointing**:
   ```bash
   torchrun --nproc_per_node=1 train_distillation.py \
     --gradient_checkpointing \
     ... other args ...
   ```

5. **Combine optimizations**:
   ```bash
   torchrun --nproc_per_node=1 train_distillation.py \
     --teacher_on_cpu \
     --teacher_dtype float16 \
     --batch_size 1 \
     --gradient_checkpointing \
     ... other args ...
   ```

---

## What Was Fixed

The following improvements were made to `train_distillation.py`:

### 1. Enhanced Error Visibility
- **Changed all error messages to use `stderr`** instead of `stdout`
  - This ensures error messages are visible even when child processes fail
  - stderr is typically not buffered, so messages appear immediately

### 2. Added Try-Catch Around Critical Operations
- **`dist.init_process_group()`** now has comprehensive error handling
  - Catches initialization failures
  - Provides detailed error messages with environment variable details
  - Suggests specific solutions based on the error

### 3. Early CUDA Validation
- **Check CUDA availability** before attempting to initialize NCCL backend
  - Prevents confusing NCCL errors when CUDA is not available
  - Clear error message with actionable solutions

### 4. GPU Device Setting Error Handling
- **Catch failures in `torch.cuda.set_device()`**
  - Happens when GPU is not accessible or busy
  - Provides clear error message with troubleshooting steps

### 5. Improved Error Messages
- All error messages now include:
  - Clear identification of the error type
  - Rank information for distributed training
  - Detailed explanation of what went wrong
  - Specific, actionable solutions
  - Environment variable values when relevant

### 6. Better Error Propagation
- Rank information included in all error messages
- Error messages from all ranks are visible
- Proper cleanup of distributed resources before exit

---

## Testing Your Setup

### Test 1: Single Process (No Distributed)
```bash
# Should work without CUDA
python train_distillation.py --help
```

### Test 2: Single GPU Distributed
```bash
# Test with one GPU
torchrun --nproc_per_node=1 train_distillation.py \
  --teacher_path "Wan-AI/Wan2.2-T2V-A14B-Diffusers" \
  --student_config "config/student_config.json" \
  --data_path "data/static_prompts.txt" \
  --distributed
```

### Test 3: Multi-GPU Distributed
```bash
# Test with multiple GPUs (adjust nproc_per_node to match your GPUs)
torchrun --nproc_per_node=2 train_distillation.py \
  --teacher_path "Wan-AI/Wan2.2-T2V-A14B-Diffusers" \
  --student_config "config/student_config.json" \
  --data_path "data/static_prompts.txt" \
  --distributed
```

---

## Debugging Tips

### 1. Check Your Environment
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Check NCCL availability
python -c "import torch; print(f'NCCL available: {torch.cuda.nccl.is_available()}')"

# Check PyTorch version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### 2. Test Simple Distributed Program
```python
# test_dist.py
import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl', init_method='env://')
print(f"Rank {dist.get_rank()} initialized successfully!")
dist.destroy_process_group()
```

```bash
torchrun --nproc_per_node=2 test_dist.py
```

### 3. Monitor GPU Usage
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

### 4. Check Logs
- Error messages are now printed to stderr
- Look for messages with `[Rank X]` prefix
- Each rank will report its own errors

---

## Summary

The key improvements:
1. ✅ All error messages now use `stderr` for better visibility
2. ✅ Comprehensive error handling around distributed initialization
3. ✅ Clear, actionable error messages with specific solutions
4. ✅ Early validation to catch common mistakes before they cause confusing errors
5. ✅ Detailed environment information in error messages
6. ✅ Rank information in all distributed error messages

These changes ensure that when `train_distillation.py` fails with torchrun, you'll get a clear, actionable error message explaining exactly what went wrong and how to fix it.
