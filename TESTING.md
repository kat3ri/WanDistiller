# Production Training Test

This document describes how to run the production training test for WanDistiller.

## Overview

The `run_production_test.py` script validates that the entire training pipeline works correctly with mock/real data. It tests:

1. ✅ Data loading from `data/static_prompts.txt`
2. ✅ Model initialization (both student and teacher)
3. ✅ Training loop execution
4. ✅ Loss computation and backpropagation
5. ✅ Model saving functionality

## Quick Start

### Basic Test (with mocked teacher model)

This is the fastest way to test the pipeline without downloading large model weights:

```bash
python run_production_test.py
```

This will:
- Use mocked teacher model (no download required)
- Load prompts from `data/static_prompts.txt` (55 prompts)
- Run 2 training epochs with batch size 2
- Save the trained model to a temporary directory

### With Custom Parameters

```bash
python run_production_test.py --num-epochs 5 --batch-size 4
```

### With Real Teacher Model

If you have access to the actual Wan model weights:

```bash
python run_production_test.py --use-real-teacher
```

⚠️ **Note**: This requires downloading the actual model weights and may need significant disk space and memory.

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--use-mock-teacher` | Use a mocked teacher model (fast, no download) | True |
| `--use-real-teacher` | Use real teacher model weights (slow, requires download) | False |
| `--num-epochs` | Number of training epochs | 2 |
| `--batch-size` | Batch size for training | 2 |

## Expected Output

When the test runs successfully, you should see output like:

```
================================================================================
WanDistiller Production Training Test
================================================================================

Test Configuration:
  - Data path: data/static_prompts.txt
  - Config path: config/student_config.json
  - Output dir: /tmp/wandistiller_test_xyz123
  - Batch size: 2
  - Epochs: 2
  - Use mock teacher: True

✓ Found 55 prompts in data file
✓ Config file found
✓ Config loaded successfully
  - Model type: WanLiteStudent
  - Hidden size: 1024
  - Depth: 16

--------------------------------------------------------------------------------
Starting Training Test
--------------------------------------------------------------------------------
Using device: cuda

1. Initializing dataset...
   ✓ Dataset created with 55 samples
   ✓ DataLoader created (batch_size=2)

2. Initializing student model...
   ✓ Student model initialized
   ✓ Total parameters: 123,456,789
   ✓ Trainable parameters: 123,456,789

3. Initializing teacher model...
   Using mocked teacher model...
   ✓ Teacher model ready

4. Initializing optimizer...
   ✓ Optimizer initialized (lr=1e-05)

5. Running training for 2 epochs...

   Epoch 1/2
      Batch 1/28: Loss = 0.123456
      Batch 8/28: Loss = 0.098765
      Batch 15/28: Loss = 0.087654
      Batch 22/28: Loss = 0.076543
   ✓ Epoch 1 complete. Average loss: 0.091505

   Epoch 2/2
      Batch 1/28: Loss = 0.065432
      Batch 8/28: Loss = 0.054321
      Batch 15/28: Loss = 0.043210
      Batch 22/28: Loss = 0.032109
   ✓ Epoch 2 complete. Average loss: 0.048768

   ✓ Training completed successfully (56 steps)

6. Testing model save...
   ✓ Model saved to /tmp/wandistiller_test_xyz123
   ✓ Saved files: student_config.json, diffusion_model.safetensors
   ✓ Config file saved
   ✓ Model weights saved

================================================================================
✓ Production Test PASSED
================================================================================

Test output saved to: /tmp/wandistiller_test_xyz123
```

## Mock Data

The test uses realistic prompts from `data/static_prompts.txt`, which contains 55 diverse prompts covering:

- 🎨 Portrait photography
- 🏔️ Landscape scenes
- 🤖 Sci-fi and cyberpunk themes
- 🏰 Fantasy and medieval settings
- 🌆 Urban and architectural scenes
- 🌸 Nature and botanical subjects
- 🎭 Cultural and historical themes
- 🍽️ Still life compositions

## Troubleshooting

### "Data file not found"
Make sure you're running the script from the repository root directory:
```bash
cd /path/to/WanDistiller
python run_production_test.py
```

### Out of Memory
Try reducing the batch size:
```bash
python run_production_test.py --batch-size 1
```

### CUDA Errors
The script will automatically fall back to CPU if CUDA is not available.

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Next Steps

After the production test passes, you can:

1. **Run full training** with more epochs:
   ```bash
   python train_distillation.py \
       --teacher_path "timbrooks/instruct-wan" \
       --student_config "config/student_config.json" \
       --data_path "data/static_prompts.txt" \
       --output_dir "./outputs/wan_t2i" \
       --num_epochs 100 \
       --batch_size 4 \
       --lr 1e-5
   ```

2. **Add more prompts** to `data/static_prompts.txt` for better coverage

3. **Adjust model configuration** in `config/student_config.json` for different model sizes

4. **Run inference** with the trained model from the output directory
