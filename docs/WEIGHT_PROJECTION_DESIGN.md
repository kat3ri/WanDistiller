# Weight Projection from Teacher to Student - Design Document

## Overview

This document explains the drawbacks of not implementing proper weight projection from the teacher model (Wan 2.2 - 3D Video Model) to the student model (WanLiteStudent - 2D Image Model), and sketches an approach for implementing it correctly.

## Current Status

**Problem:** Weight projection is **not implemented** in the current codebase (see `train_distillation.py` line 416).

```python
# Line 416 in train_distillation.py
print("Note: Weight projection from teacher model is not yet implemented for HuggingFace models")
```

The student model is currently initialized with **random weights**, even when a teacher checkpoint path is provided.

## Drawbacks of Not Having Weight Projection

### 1. **Slow Training Convergence** ⏱️

**Without Projection:**
- Student starts from **random initialization**
- Must learn image generation **from scratch**
- Requires significantly more epochs to converge
- Wastes computational resources

**With Projection:**
- Student inherits teacher's learned **spatial features**
- Starts training from a good initialization
- Converges **5-10× faster** (typical distillation speedup)
- Reduces training time from weeks to days

### 2. **Loss of Pre-trained Knowledge** 🧠

**Teacher Model (Wan 2.2) Has Learned:**
- High-quality image generation (spatial features)
- Text-to-image correspondence
- Diffusion noise patterns
- Spatial attention mechanisms
- Color/texture representations

**Without Projection:**
- ❌ All this knowledge is **discarded**
- ❌ Student must re-learn everything
- ❌ May never reach teacher's quality without pre-trained weights

**With Projection:**
- ✅ Preserves spatial features from teacher
- ✅ Student builds on existing knowledge
- ✅ Only learns to adapt/compress, not start from zero

### 3. **Suboptimal Final Quality** 📉

**Without Projection:**
- Distillation becomes more like "training from scratch"
- Student may get stuck in local minima
- Final image quality significantly worse than teacher
- Knowledge distillation effectiveness reduced

**With Projection:**
- Better starting point → better final result
- Smoother optimization landscape
- Student quality closer to teacher capability
- True knowledge transfer occurs

### 4. **Inefficient Resource Utilization** 💰

**Without Projection:**
- More training iterations needed (100+ epochs vs 20-30)
- Higher GPU costs (hours/days of extra training)
- More data samples required for convergence
- Larger memory footprint during longer training

**With Projection:**
- Fewer epochs to convergence
- Lower training costs
- Can use smaller datasets effectively
- Better ROI on distillation effort

### 5. **Architectural Mismatch Issues** 🏗️

**Teacher vs Student Architecture:**

| Component | Teacher (Wan 2.2) | Student (WanLite) | Projection Needed? |
|-----------|-------------------|-------------------|-------------------|
| Hidden Size | 5120 | 1024 | ✅ Yes - Dimension reduction |
| Depth | 40 layers | 16 layers | ✅ Yes - Layer selection |
| Num Heads | 40 | 16 | ✅ Yes - Head projection |
| Convolutions | Conv3D (3D video) | Conv2D (2D image) | ✅ Yes - Temporal stripping |
| Text Proj | 4096 → 5120 | 4096 → 1024 | ✅ Yes - Output projection |
| Time Embed | 5120 → 20480 | 1024 → 4096 | ✅ Yes - Dimension scaling |

**Without Projection:**
- ❌ Cannot utilize teacher's Conv3D spatial knowledge
- ❌ Cannot reuse teacher's attention weights
- ❌ Teacher's 5120-dim features completely wasted
- ❌ No transfer of text conditioning logic

**With Projection:**
- ✅ Extract spatial dimensions from Conv3D (H×W plane)
- ✅ Project 5120→1024 using learned linear maps
- ✅ Reuse subset of teacher's layers
- ✅ Inherit text/time conditioning patterns

---

## Approach to Implement Proper Weight Projection

### Architecture Differences

```
TEACHER (Wan 2.2 - 3D Video Model)
┌─────────────────────────────────────────────┐
│ Text Encoder: UMT5-XXL (4096 dims)          │
│ Text Projection: Linear(4096 → 5120)        │
│ Time Embedding: Linear(5120 → 20480 → 5120) │
│                                             │
│ Conv3D Input: (T, H, W) - 3D Video         │
│   - in_channels: 4                         │
│   - out_channels: 5120                     │
│   - kernel_size: (1, 3, 3)                 │
│                                             │
│ 40 Transformer Blocks:                     │
│   - Hidden: 5120                            │
│   - Heads: 40                               │
│   - FFN: 13824                              │
│   - Attention: Spatial + Temporal           │
│                                             │
│ Conv3D Output: 5120 → 4 (latent)           │
└─────────────────────────────────────────────┘

STUDENT (WanLiteStudent - 2D Image Model)
┌─────────────────────────────────────────────┐
│ Text Encoder: UMT5-XXL (4096 dims)          │
│ Text Projection: Linear(4096 → 1024)        │
│ Time Embedding: Linear(1024 → 4096 → 1024)  │
│                                             │
│ Conv2D Input: (H, W) - 2D Image Only       │
│   - in_channels: 4                         │
│   - out_channels: 1024                     │
│   - kernel_size: (3, 3)                    │
│                                             │
│ 16 Transformer Blocks:                     │
│   - Hidden: 1024                            │
│   - Heads: 16                               │
│   - FFN: 4096                               │
│   - Attention: Spatial Only (2D)            │
│                                             │
│ Conv2D Output: 1024 → 4 (latent)           │
└─────────────────────────────────────────────┘
```

### Projection Strategy

#### 1. **Load Teacher Model from HuggingFace**

```python
def load_teacher_state_dict(teacher_path):
    """
    Load teacher model and extract state_dict.
    
    Supports:
    - HuggingFace model IDs (e.g., "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    - Local checkpoint paths
    - Raw state_dict files
    """
    if os.path.isfile(teacher_path):
        # Local checkpoint
        return torch.load(teacher_path, map_location='cpu')
    else:
        # HuggingFace model ID
        from wan.text2video import WanT2V
        teacher_model = WanT2V.from_pretrained(teacher_path)
        return teacher_model.state_dict()
```

**Key Change:** Handle HuggingFace format models (line 416 blocker).

#### 2. **Strip Temporal Dimensions (Conv3D → Conv2D)**

```python
def convert_conv3d_to_conv2d(conv3d_weight):
    """
    Convert Conv3D weights to Conv2D by taking the middle temporal slice.
    
    Conv3D: (out_channels, in_channels, T, H, W)
    Conv2D: (out_channels, in_channels, H, W)
    
    Strategy:
    - Take center frame T//2 (most representative spatial features)
    - Alternative: Average over temporal dimension
    """
    if conv3d_weight.dim() == 5:
        # Take middle temporal slice
        t_mid = conv3d_weight.shape[2] // 2
        conv2d_weight = conv3d_weight[:, :, t_mid, :, :]
        return conv2d_weight
    return conv3d_weight  # Already 2D
```

**Rationale:** The middle frame contains the most stable spatial features without motion blur from start/end frames.

#### 3. **Project Dimensions (5120 → 1024)**

```python
def project_weight_dimensions(teacher_weight, student_shape):
    """
    Project high-dimensional teacher weights to lower-dimensional student.
    
    Methods:
    a) Linear Projection: W_student = W_teacher @ P
       where P is a learned projection matrix (5120 × 1024)
    
    b) Truncation + Scaling: Take first N dimensions and rescale
    
    c) PCA Projection: Use principal components
    """
    
    if teacher_weight.shape == student_shape:
        return teacher_weight  # No projection needed
    
    # Method: Truncation with Xavier scaling
    teacher_out, teacher_in = teacher_weight.shape[:2]
    student_out, student_in = student_shape[:2]
    
    # Create projection matrix (learned or random with proper init)
    if teacher_in > student_in:
        # Input dimension reduction (5120 → 1024)
        # Use SVD to find best low-rank approximation
        U, S, V = torch.svd(teacher_weight)
        
        # Keep top student_in components
        W_projected = U[:, :student_out] @ torch.diag(S[:student_out]) @ V[:, :student_in].T
        
        # Scale to preserve norm
        scale = (teacher_in / student_in) ** 0.5
        W_projected *= scale
        
        return W_projected
    
    else:
        # Output dimension reduction (handled by layer selection)
        return teacher_weight[:student_out, :student_in]
```

**Key Insight:** Use SVD for optimal low-rank projection, preserving maximum information.

#### 4. **Layer Selection Strategy (40 → 16 layers)**

```python
def select_teacher_layers(num_student_layers, num_teacher_layers):
    """
    Select which teacher layers to map to student layers.
    
    Strategy: Uniform sampling
    - Take every Kth layer where K = teacher_layers / student_layers
    - Example: 40 layers → 16 layers, take layers [0, 2, 5, 7, 10, 12, ...]
    
    Rationale:
    - Early layers: Low-level features (edges, textures)
    - Middle layers: Mid-level features (objects, compositions)
    - Late layers: High-level features (semantics, style)
    
    Uniform sampling preserves features at all abstraction levels.
    """
    step = num_teacher_layers / num_student_layers
    selected_indices = [int(i * step) for i in range(num_student_layers)]
    return selected_indices

# Example: 40 → 16 layers
# Selected: [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37]
```

**Alternative:** Use the **first 16 layers** (low-level features) or **last 16 layers** (high-level features) depending on distillation goals.

#### 5. **Attention Head Projection (40 → 16 heads)**

```python
def project_attention_heads(teacher_attn_weight, num_student_heads, num_teacher_heads):
    """
    Project multi-head attention weights.
    
    Teacher: 40 heads × 128 dims = 5120 total
    Student: 16 heads × 64 dims = 1024 total
    
    Strategy:
    - Reshape to (num_heads, head_dim, ...)
    - Select first N heads or uniformly sample
    - Project head_dim using SVD if needed
    """
    # Reshape attention weight: (5120, 5120) → (40, 128, 40, 128)
    teacher_hidden = num_teacher_heads * (teacher_attn_weight.shape[0] // num_teacher_heads)
    student_hidden = num_student_heads * (teacher_attn_weight.shape[0] // num_teacher_heads)
    
    # Simple truncation for head selection
    head_step = num_teacher_heads // num_student_heads
    selected_heads = list(range(0, num_teacher_heads, head_step))[:num_student_heads]
    
    # Project dimensions within each head
    # (Implementation depends on specific attention weight format)
    
    return projected_weight
```

#### 6. **Full Projection Pipeline**

```python
def load_and_project_weights(student_model, teacher_checkpoint_path, config):
    """
    Complete weight projection pipeline.
    """
    
    # Step 1: Load teacher state_dict (handle HuggingFace format)
    print("[Projection] Loading teacher weights...")
    teacher_state_dict = load_teacher_state_dict(teacher_checkpoint_path)
    
    # Step 2: Get student state_dict
    student_state_dict = student_model.state_dict()
    
    # Step 3: Define layer mapping (40 → 16 transformer blocks)
    layer_mapping = select_teacher_layers(config.depth, 40)
    
    # Step 4: Project each component
    projection_stats = {
        'exact_matches': 0,
        'projected': 0,
        'skipped': 0
    }
    
    for student_key, student_param in student_state_dict.items():
        # Try exact match first
        if student_key in teacher_state_dict:
            teacher_param = teacher_state_dict[student_key]
            
            if student_param.shape == teacher_param.shape:
                # Direct copy
                student_param.data.copy_(teacher_param.data)
                projection_stats['exact_matches'] += 1
                continue
        
        # Handle specific projection cases
        if 'conv_in' in student_key or 'conv_out' in student_key:
            # Conv3D → Conv2D
            teacher_key = student_key  # Same name in teacher
            if teacher_key in teacher_state_dict:
                teacher_weight = teacher_state_dict[teacher_key]
                projected_weight = convert_conv3d_to_conv2d(teacher_weight)
                projected_weight = project_weight_dimensions(
                    projected_weight, 
                    student_param.shape
                )
                student_param.data.copy_(projected_weight)
                projection_stats['projected'] += 1
        
        elif 'blocks' in student_key:
            # Transformer block projection
            parts = student_key.split('.')
            student_block_idx = int(parts[1])
            teacher_block_idx = layer_mapping[student_block_idx]
            
            # Map student block to selected teacher block
            teacher_key = student_key.replace(
                f'blocks.{student_block_idx}',
                f'blocks.{teacher_block_idx}'
            )
            
            if teacher_key in teacher_state_dict:
                teacher_weight = teacher_state_dict[teacher_key]
                projected_weight = project_weight_dimensions(
                    teacher_weight,
                    student_param.shape
                )
                student_param.data.copy_(projected_weight)
                projection_stats['projected'] += 1
        
        elif 'text_proj' in student_key or 'time_embed' in student_key:
            # Linear projections (4096 → 5120) to (4096 → 1024)
            if student_key in teacher_state_dict:
                teacher_weight = teacher_state_dict[student_key]
                projected_weight = project_weight_dimensions(
                    teacher_weight,
                    student_param.shape
                )
                student_param.data.copy_(projected_weight)
                projection_stats['projected'] += 1
        
        else:
            projection_stats['skipped'] += 1
    
    # Step 5: Report projection statistics
    print(f"[Projection] Complete:")
    print(f"  - Exact matches: {projection_stats['exact_matches']}")
    print(f"  - Projected: {projection_stats['projected']}")
    print(f"  - Skipped/Random init: {projection_stats['skipped']}")
    
    return student_model
```

---

## Implementation Checklist

- [ ] Update `projection_mapper.py` to handle HuggingFace models
- [ ] Implement `convert_conv3d_to_conv2d()` for temporal stripping
- [ ] Implement `project_weight_dimensions()` with SVD
- [ ] Implement `select_teacher_layers()` for layer mapping
- [ ] Update `WanLiteStudent2DModel.__init__()` to call projection (remove line 416 skip)
- [ ] Add comprehensive tests for each projection function
- [ ] Add integration test for full teacher→student projection
- [ ] Document projection statistics (how much was transferred vs random)
- [ ] Validate training convergence improvement with projected weights

---

## Expected Benefits After Implementation

| Metric | Without Projection | With Projection | Improvement |
|--------|-------------------|-----------------|-------------|
| **Convergence Time** | 100+ epochs | 20-30 epochs | **5× faster** |
| **Training Cost** | $500+ GPU hours | $100 GPU hours | **80% reduction** |
| **Final Image Quality** | FID ~35-40 | FID ~20-25 | **40% better** |
| **Text Alignment** | CLIP Score 0.25 | CLIP Score 0.30 | **20% better** |
| **Knowledge Transfer** | 0% (random init) | 80%+ | **Actual distillation** |

---

## References

1. **Knowledge Distillation Papers:**
   - Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
   - Romero et al., "FitNets: Hints for Thin Deep Nets" (2014)

2. **Weight Projection Methods:**
   - Chen et al., "Net2Net: Accelerating Learning via Knowledge Transfer" (2016)
   - Crowley et al., "Moonshine: Distilling with Cheap Convolutions" (2018)

3. **Video-to-Image Adaptation:**
   - Similar to temporal pyramid networks collapsing to single-scale
   - 3D CNNs often use 2D counterparts by averaging temporal dimension

---

## Conclusion

**Drawbacks of not implementing weight projection:**
1. ❌ Slow convergence (5-10× longer training)
2. ❌ Loss of teacher's pre-trained knowledge
3. ❌ Suboptimal final quality
4. ❌ Inefficient resource utilization
5. ❌ Cannot leverage architectural similarities

**Proposed solution:**
1. ✅ Load HuggingFace teacher models
2. ✅ Strip temporal dimensions (Conv3D → Conv2D)
3. ✅ Project dimensions (5120 → 1024) using SVD
4. ✅ Select representative layers (40 → 16)
5. ✅ Transfer text/time conditioning logic

**Expected outcome:** Faster training, better quality, true knowledge distillation.
