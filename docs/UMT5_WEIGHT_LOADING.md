# UMT5 Text Encoder Weight Loading

## Issue

When loading the Wan2.2-T2V model with a UMT5EncoderModel text encoder, you may see a warning message like:

```
UMT5EncoderModel LOAD REPORT from: ...text_encoder
[Deserializing param=shared.weight] 
Key                         | Status  | Details
----------------------------+---------+--------
encoder.embed_tokens.weight | MISSING |

Notes:
- MISSING: those params were newly initialized because missing from the checkpoint.
```

## Explanation

**This warning is expected and harmless.** It does not indicate a problem with the model.

### Why This Happens

T5 and UMT5 models use **weight tying** to share embeddings across different parts of the model:

- `shared.weight` - The actual parameter containing the embedding weights
- `encoder.embed_tokens.weight` - Tied to `shared.weight` (references the same tensor)
- `decoder.embed_tokens.weight` - Also tied to `shared.weight` (in full T5 models)

When the model is saved, only `shared.weight` is serialized to avoid duplication. During loading:

1. `shared.weight` is loaded from the checkpoint ✓
2. `encoder.embed_tokens.weight` is not found in the checkpoint (appears as "MISSING")
3. However, the model automatically ties `encoder.embed_tokens.weight` to `shared.weight`
4. The "newly initialized" message is misleading - the weights are actually loaded via the tie

### Impact

- **No impact on model functionality** - The embeddings are correctly loaded via `shared.weight`
- **No need to retrain** - Despite the warning message suggesting training may be needed
- **Normal T5/UMT5 behavior** - This is how these models are designed to work

## Solution

The warning can be safely ignored. However, if you want to suppress it, the `train_distillation.py` script now includes:

```python
import warnings

# When loading the teacher model
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*were newly initialized.*")
    teacher_pipe = DiffusionPipeline.from_pretrained(model_path)
```

## Verification

To verify that weights are correctly loaded, you can check:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

# These should reference the same tensor
assert pipe.text_encoder.shared.weight is pipe.text_encoder.encoder.embed_tokens.weight
print("✓ Weights are correctly tied")
```

## References

- [Hugging Face Transformers: T5 Model](https://huggingface.co/docs/transformers/model_doc/t5)
- [Weight Tying in Language Models](https://paperswithcode.com/method/weight-tying)
