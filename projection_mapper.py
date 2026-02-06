import torch
import torch.nn as nn
import sys
import json
import os
from pathlib import Path

# Assuming student_config is imported or path is handled correctly
try:
    # Adjust path as necessary based on your project structure
    sys.path.append(str(Path(__file__).resolve().parent))
    from student_config import StudentConfig
except ImportError:
    # Fallback config if file not found during standalone testing
    class StudentConfig:
        hidden_size = 640
        num_heads = 10


def load_teacher_state_dict(teacher_checkpoint_path):
    """
    Load teacher model state_dict from various sources.
    
    Supports:
    - HuggingFace model IDs (e.g., "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    - Local checkpoint files (.pt, .pth, .safetensors)
    - Direct state_dict objects
    
    Args:
        teacher_checkpoint_path: Path or model ID or state_dict
        
    Returns:
        state_dict: Teacher model state dictionary
    """
    # If already a state_dict, return as-is
    if isinstance(teacher_checkpoint_path, dict):
        return teacher_checkpoint_path
    
    # If local file, load it
    if isinstance(teacher_checkpoint_path, (str, Path)) and os.path.isfile(str(teacher_checkpoint_path)):
        print(f"[Projection] Loading teacher from local file: {teacher_checkpoint_path}")
        if str(teacher_checkpoint_path).endswith('.safetensors'):
            from safetensors.torch import load_file
            return load_file(teacher_checkpoint_path)
        else:
            checkpoint = torch.load(teacher_checkpoint_path, map_location='cpu')
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                return checkpoint['state_dict']
            elif 'model' in checkpoint:
                return checkpoint['model']
            else:
                return checkpoint
    
    # Try to load as HuggingFace model or WAN directory
    print(f"[Projection] Attempting to load teacher from directory: {teacher_checkpoint_path}")
    try:
        # Import here to avoid dependency if not needed
        from wan.text2video import WanT2V, _is_huggingface_format
        from wan.modules.model import WanModel
        
        # Check if it's a directory with WAN models
        if isinstance(teacher_checkpoint_path, (str, Path)) and os.path.isdir(str(teacher_checkpoint_path)):
            is_hf_format = _is_huggingface_format(teacher_checkpoint_path)
            print(f"[Projection] Directory format detected: {'HuggingFace' if is_hf_format else 'Local'}")
            
            # Try to load the transformer model directly using WanModel
            if is_hf_format:
                # HuggingFace format: load from transformer subfolder
                subfolder = 'transformer'
            else:
                # Local format: try low_noise_model subfolder
                subfolder = 'low_noise_model'
            
            try:
                print(f"[Projection] Loading WanModel from subfolder: {subfolder}")
                teacher_model = WanModel.from_pretrained(teacher_checkpoint_path, subfolder=subfolder)
                state_dict = teacher_model.state_dict()
                print(f"[Projection] Successfully loaded {len(state_dict)} keys from teacher model")
                return state_dict
            except Exception as e:
                print(f"[Projection] Could not load from {subfolder}: {type(e).__name__}: {e}")
                # Try alternate subfolder
                alt_subfolder = 'transformer_2' if is_hf_format else 'high_noise_model'
                try:
                    print(f"[Projection] Trying alternate subfolder: {alt_subfolder}")
                    teacher_model = WanModel.from_pretrained(teacher_checkpoint_path, subfolder=alt_subfolder)
                    state_dict = teacher_model.state_dict()
                    print(f"[Projection] Successfully loaded {len(state_dict)} keys from teacher model")
                    return state_dict
                except Exception as e2:
                    print(f"[Projection] Could not load from {alt_subfolder}: {type(e2).__name__}: {e2}")
        
        print(f"[Projection] ERROR: Could not load teacher model from any known format")
        print(f"[Projection] Path attempted: {teacher_checkpoint_path}")
        print(f"[Projection] WARNING: Returning empty state_dict - training will use random initialization")
        return {}
    except Exception as e:
        print(f"[Projection] ERROR: Exception while loading teacher model")
        print(f"[Projection] Exception type: {type(e).__name__}")
        print(f"[Projection] Exception message: {e}")
        print(f"[Projection] WARNING: Returning empty state_dict - training will use random initialization")
        return {}


def convert_conv3d_to_conv2d(conv3d_weight):
    """
    Convert Conv3D weights to Conv2D by taking the middle temporal slice.
    
    Conv3D: (out_channels, in_channels, T, H, W)
    Conv2D: (out_channels, in_channels, H, W)
    
    Strategy: Take center frame T//2 (most representative spatial features)
    
    Args:
        conv3d_weight: 5D tensor (out_channels, in_channels, T, H, W)
        
    Returns:
        conv2d_weight: 4D tensor (out_channels, in_channels, H, W)
    """
    if conv3d_weight.dim() == 5:
        # Take middle temporal slice
        t_mid = conv3d_weight.shape[2] // 2
        conv2d_weight = conv3d_weight[:, :, t_mid, :, :]
        print(f"[Projection] Converted Conv3D {conv3d_weight.shape} → Conv2D {conv2d_weight.shape}")
        return conv2d_weight
    elif conv3d_weight.dim() == 4:
        # Already Conv2D
        return conv3d_weight
    else:
        print(f"[Projection] Warning: Unexpected conv weight shape {conv3d_weight.shape}")
        return conv3d_weight


def project_weight_dimensions(teacher_weight, student_shape, method='truncate'):
    """
    Project high-dimensional teacher weights to lower-dimensional student.
    
    Supports multiple projection methods:
    - 'truncate': Take first N dimensions with scaling (fast, simple)
    - 'svd': Use SVD for optimal low-rank approximation (better quality, slower)
    - 'average': Average groups of dimensions (preserves more information)
    
    Args:
        teacher_weight: Teacher weight tensor
        student_shape: Target student weight shape
        method: Projection method ('truncate', 'svd', 'average')
        
    Returns:
        projected_weight: Projected weight matching student_shape
    """
    if teacher_weight.shape == student_shape:
        return teacher_weight  # No projection needed
    
    # Handle linear layers (2D)
    if len(teacher_weight.shape) == 2 and len(student_shape) == 2:
        teacher_out, teacher_in = teacher_weight.shape
        student_out, student_in = student_shape
        
        if method == 'truncate':
            # Simple truncation with scaling
            projected = teacher_weight[:student_out, :student_in].clone()
            # Scale to preserve variance
            scale = (teacher_in / student_in) ** 0.5 if teacher_in > student_in else 1.0
            projected *= scale
            return projected
            
        elif method == 'average':
            # Average pooling for dimension reduction
            if teacher_in > student_in:
                # Group teacher dims and average
                group_size = teacher_in // student_in
                projected_in = []
                for i in range(student_in):
                    start_idx = i * group_size
                    end_idx = (i + 1) * group_size if i < student_in - 1 else teacher_in
                    projected_in.append(teacher_weight[:, start_idx:end_idx].mean(dim=1, keepdim=True))
                projected = torch.cat(projected_in, dim=1)
                projected = projected[:student_out, :]
                return projected
            else:
                return teacher_weight[:student_out, :student_in]
                
        elif method == 'svd':
            # SVD-based projection (best quality but slower)
            try:
                # Ensure teacher_weight is on CPU and float32 for SVD
                tw = teacher_weight.cpu().float()
                U, S, V = torch.svd(tw)
                
                # Reconstruct with reduced dimensions
                k = min(student_out, student_in, U.shape[1], V.shape[0])
                projected = U[:, :k] @ torch.diag(S[:k]) @ V[:, :k].T
                
                # Take only needed dimensions
                projected = projected[:student_out, :student_in]
                
                # Restore original dtype and device
                projected = projected.to(dtype=teacher_weight.dtype, device=teacher_weight.device)
                return projected
            except Exception as e:
                print(f"[Projection] SVD failed: {e}, falling back to truncate")
                return project_weight_dimensions(teacher_weight, student_shape, method='truncate')
    
    # Handle Conv2D layers (4D)
    elif len(teacher_weight.shape) == 4 and len(student_shape) == 4:
        # (out_channels, in_channels, H, W)
        teacher_out, teacher_in, kh_t, kw_t = teacher_weight.shape
        student_out, student_in, kh_s, kw_s = student_shape
        
        # Project channels
        projected = teacher_weight[:student_out, :student_in, :, :]
        
        # Handle kernel size mismatch if needed
        if (kh_t, kw_t) != (kh_s, kw_s):
            # Center crop or pad kernel
            if kh_t >= kh_s and kw_t >= kw_s:
                # Crop to center
                h_start = (kh_t - kh_s) // 2
                w_start = (kw_t - kw_s) // 2
                projected = projected[:, :, h_start:h_start+kh_s, w_start:w_start+kw_s]
            else:
                # Pad with zeros
                pad_h = (kh_s - kh_t) // 2
                pad_w = (kw_s - kw_t) // 2
                projected = torch.nn.functional.pad(
                    projected, 
                    (pad_w, kw_s - kw_t - pad_w, pad_h, kh_s - kh_t - pad_h)
                )
        
        # Scale to preserve variance
        scale = (teacher_in / student_in) ** 0.5 if teacher_in > student_in else 1.0
        projected *= scale
        
        return projected
    
    # Fallback: just truncate dimensions
    print(f"[Projection] Warning: Using fallback truncation for shape {teacher_weight.shape} → {student_shape}")
    slices = tuple(slice(0, min(t, s)) for t, s in zip(teacher_weight.shape, student_shape))
    projected = teacher_weight[slices].clone()
    
    # Pad if needed
    if projected.shape != student_shape:
        pad_amounts = []
        for t, s in zip(reversed(projected.shape), reversed(student_shape)):
            pad_amounts.extend([0, s - t])
        if any(p > 0 for p in pad_amounts):
            projected = torch.nn.functional.pad(projected, pad_amounts)
    
    return projected


def select_teacher_layers(num_student_layers, num_teacher_layers=40):
    """
    Select which teacher layers to map to student layers.
    
    Strategy: Uniform sampling to preserve features at all abstraction levels.
    
    Args:
        num_student_layers: Number of layers in student (e.g., 16)
        num_teacher_layers: Number of layers in teacher (e.g., 40)
        
    Returns:
        List of teacher layer indices to use
        
    Example:
        40 → 16 layers: [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37]
    """
    if num_student_layers >= num_teacher_layers:
        # Student has more layers, use all teacher layers
        return list(range(num_teacher_layers))
    
    step = num_teacher_layers / num_student_layers
    selected_indices = [int(i * step) for i in range(num_student_layers)]
    
    print(f"[Projection] Layer mapping ({num_teacher_layers} → {num_student_layers}):")
    print(f"  Selected teacher layers: {selected_indices}")
    
    return selected_indices


def map_teacher_to_student_key(student_key, student_block_idx, layer_mapping):
    """
    Map a student parameter key to the corresponding teacher parameter key.
    
    Handles the architectural differences between teacher (WanAttentionBlock) and 
    student (WanTransformerBlock):
    - Teacher uses self_attn with separate q, k, v, o projections
    - Student uses attn (nn.MultiheadAttention) with in_proj (combined q,k,v) and out_proj
    
    Args:
        student_key: Student parameter key (e.g., 'blocks.0.attn.in_proj_weight')
        student_block_idx: Student block index
        layer_mapping: List mapping student layers to teacher layers
        
    Returns:
        List of (teacher_key, slice_info) tuples for mapping teacher params to student.
        slice_info is None for direct copy, or (dim, start, end) for slicing.
    """
    # Map student block index to teacher block index
    teacher_block_idx = layer_mapping[student_block_idx] if student_block_idx < len(layer_mapping) else student_block_idx
    
    # Handle MultiheadAttention parameter mapping
    # Student uses nn.MultiheadAttention which has in_proj_weight (concatenated q,k,v) and out_proj
    # Teacher uses separate q, k, v, o linear layers
    
    if '.attn.in_proj_weight' in student_key:
        # Student's in_proj_weight = [q_weight; k_weight; v_weight] concatenated
        # Map to teacher's self_attn.q, self_attn.k, self_attn.v
        base_key = f'blocks.{teacher_block_idx}.self_attn'
        return [
            (f'{base_key}.q.weight', ('cat', 0)),  # Concatenate along dim 0
            (f'{base_key}.k.weight', ('cat', 0)),
            (f'{base_key}.v.weight', ('cat', 0)),
        ]
    elif '.attn.in_proj_bias' in student_key:
        # Student's in_proj_bias = [q_bias; k_bias; v_bias] concatenated
        base_key = f'blocks.{teacher_block_idx}.self_attn'
        return [
            (f'{base_key}.q.bias', ('cat', 0)),
            (f'{base_key}.k.bias', ('cat', 0)),
            (f'{base_key}.v.bias', ('cat', 0)),
        ]
    elif '.attn.out_proj.weight' in student_key:
        # Direct mapping
        return [(f'blocks.{teacher_block_idx}.self_attn.o.weight', None)]
    elif '.attn.out_proj.bias' in student_key:
        # Direct mapping
        return [(f'blocks.{teacher_block_idx}.self_attn.o.bias', None)]
    else:
        # For other parameters (norm1, norm2, mlp), try direct mapping with teacher block index
        teacher_key = student_key.replace(f'blocks.{student_block_idx}.', f'blocks.{teacher_block_idx}.')
        return [(teacher_key, None)]


def load_and_project_weights(student_model, teacher_checkpoint_path, config=None, device="cuda", projection_method='truncate'):
    """
    Loads weights from Teacher (Wan 2.2 - 3D Video Model) to Student (2D Image Model).
    
    This function handles the conversion from a 3D video generation model to a 2D
    image generation model by:
    1. Loading teacher weights from HuggingFace or local checkpoint
    2. Stripping temporal dimensions from the teacher model (Conv3D → Conv2D)
    3. Projecting weights when dimensions don't match (5120 → 1024)
    4. Mapping teacher layers to student layers (40 → 16)
    5. Handling attention module differences (self_attn.q/k/v/o → attn.in_proj/out_proj)
    6. Intelligently initializing weights for size mismatches
    
    The teacher model has temporal/motion components that are removed in the student.
    The student is purely spatial (2D) for static image generation.
    
    Args:
        student_model: The 2D student model to initialize
        teacher_checkpoint_path: Path to teacher checkpoint, HF model ID, or state_dict
        config: Configuration with student architecture details
        device: Device to place weights on
        projection_method: Method for dimension projection ('truncate', 'svd', 'average')
        
    Returns:
        student_model: Student model with projected weights loaded
    """
    if config is None:
        config = StudentConfig()

    # Step 1: Load teacher state_dict (handle HuggingFace format)
    print(f"[Projection] ================================================================================")
    print(f"[Projection] Starting Weight Projection from Teacher to Student")
    print(f"[Projection] ================================================================================")
    
    teacher_state_dict = load_teacher_state_dict(teacher_checkpoint_path)
    
    if not teacher_state_dict:
        print(f"[Projection] ================================================================================")
        print(f"[Projection] WARNING: Empty teacher state_dict - NO WEIGHT TRANSFER WILL OCCUR")
        print(f"[Projection] ================================================================================")
        print(f"[Projection] The student model will be trained from random initialization.")
        print(f"[Projection] This may significantly impact model quality and training time.")
        print(f"[Projection] Please verify the teacher checkpoint path: {teacher_checkpoint_path}")
        print(f"[Projection] ================================================================================")
        return student_model
    
    student_state_dict = student_model.state_dict()

    print(f"[Projection] Student Config:")
    print(f"  - Hidden Size: {config.hidden_size if hasattr(config, 'hidden_size') else 'N/A'}")
    print(f"  - Depth: {config.depth if hasattr(config, 'depth') else 'N/A'}")
    print(f"  - Num Heads: {config.num_heads if hasattr(config, 'num_heads') else 'N/A'}")
    print(f"[Projection] Teacher State Dict: {len(teacher_state_dict)} keys")
    print(f"[Projection] Student State Dict: {len(student_state_dict)} keys")
    print(f"[Projection] Projection Method: {projection_method}")
    print()

    # Step 2: Define layer mapping (if teacher has more layers than student)
    num_student_layers = config.depth if hasattr(config, 'depth') else 16
    num_teacher_layers = 40  # Wan 2.2 has 40 layers
    layer_mapping = select_teacher_layers(num_student_layers, num_teacher_layers)
    
    # Step 3: Project each component
    projection_stats = {
        'exact_matches': 0,
        'projected': 0,
        'skipped': 0,
        'conv3d_to_2d': 0
    }

    # Iterate over student parameters
    for student_key, student_param in student_state_dict.items():
        transferred = False
        
        # --- Try exact match first (for non-block parameters like text_proj, time_embed, etc.) ---
        if student_key in teacher_state_dict:
            teacher_param = teacher_state_dict[student_key]
            
            # Check for Conv3D → Conv2D conversion
            if teacher_param.dim() == 5 and student_param.dim() == 4:
                # Convert Conv3D to Conv2D
                teacher_param = convert_conv3d_to_conv2d(teacher_param)
                projection_stats['conv3d_to_2d'] += 1
            
            if student_param.shape == teacher_param.shape:
                # Direct copy
                student_param.data.copy_(teacher_param.data)
                projection_stats['exact_matches'] += 1
                transferred = True
            else:
                # Dimension mismatch - project
                try:
                    projected = project_weight_dimensions(
                        teacher_param,
                        student_param.shape,
                        method=projection_method
                    )
                    student_param.data.copy_(projected.to(student_param.device))
                    projection_stats['projected'] += 1
                    print(f"[Projection] Projected {student_key}: {teacher_param.shape} → {student_param.shape}")
                    transferred = True
                except Exception as e:
                    print(f"[Projection] Error projecting {student_key}: {e}")
                    projection_stats['skipped'] += 1
        
        # --- Handle transformer blocks with layer mapping and key translation ---
        if not transferred and 'blocks.' in student_key:
            parts = student_key.split('.')
            if len(parts) >= 2 and parts[0] == 'blocks':
                try:
                    student_block_idx = int(parts[1])
                    
                    # Get mapped teacher keys for this student key
                    teacher_mappings = map_teacher_to_student_key(student_key, student_block_idx, layer_mapping)
                    
                    if teacher_mappings:
                        # Check if this is a concatenated mapping (e.g., in_proj from q,k,v)
                        if any(mapping[1] and mapping[1][0] == 'cat' for mapping in teacher_mappings):
                            # Handle concatenation of multiple teacher params
                            teacher_params = []
                            all_found = True
                            for teacher_key, slice_info in teacher_mappings:
                                if teacher_key in teacher_state_dict:
                                    teacher_param = teacher_state_dict[teacher_key]
                                    
                                    # Check for Conv3D → Conv2D conversion
                                    if teacher_param.dim() == 5 and student_param.dim() == 4:
                                        teacher_param = convert_conv3d_to_conv2d(teacher_param)
                                        projection_stats['conv3d_to_2d'] += 1
                                    
                                    teacher_params.append(teacher_param)
                                else:
                                    all_found = False
                                    break
                            
                            if all_found and teacher_params:
                                try:
                                    # Concatenate teacher params (q, k, v)
                                    cat_dim = teacher_mappings[0][1][1] if teacher_mappings[0][1] else 0
                                    concatenated = torch.cat(teacher_params, dim=cat_dim)
                                    
                                    # Project if needed
                                    if concatenated.shape == student_param.shape:
                                        student_param.data.copy_(concatenated.data)
                                        projection_stats['exact_matches'] += 1
                                        transferred = True
                                    else:
                                        projected = project_weight_dimensions(
                                            concatenated,
                                            student_param.shape,
                                            method=projection_method
                                        )
                                        student_param.data.copy_(projected.to(student_param.device))
                                        projection_stats['projected'] += 1
                                        if projection_stats['projected'] <= 5:  # Only print first few
                                            print(f"[Projection] Projected {student_key} (concat from {len(teacher_params)} params): {concatenated.shape} → {student_param.shape}")
                                        transferred = True
                                except Exception as e:
                                    print(f"[Projection] Error concatenating/projecting {student_key}: {e}")
                        else:
                            # Single teacher param mapping
                            teacher_key, _ = teacher_mappings[0]
                            if teacher_key in teacher_state_dict:
                                teacher_param = teacher_state_dict[teacher_key]
                                
                                # Check for Conv3D → Conv2D conversion
                                if teacher_param.dim() == 5 and student_param.dim() == 4:
                                    teacher_param = convert_conv3d_to_conv2d(teacher_param)
                                    projection_stats['conv3d_to_2d'] += 1
                                
                                if student_param.shape == teacher_param.shape:
                                    student_param.data.copy_(teacher_param.data)
                                    projection_stats['exact_matches'] += 1
                                    transferred = True
                                else:
                                    # Project dimensions
                                    try:
                                        projected = project_weight_dimensions(
                                            teacher_param,
                                            student_param.shape,
                                            method=projection_method
                                        )
                                        student_param.data.copy_(projected.to(student_param.device))
                                        projection_stats['projected'] += 1
                                        if projection_stats['projected'] <= 10:  # Print first few
                                            teacher_block_idx = layer_mapping[student_block_idx] if student_block_idx < len(layer_mapping) else student_block_idx
                                            print(f"[Projection] Projected {student_key} (from layer {teacher_block_idx}): {teacher_param.shape} → {student_param.shape}")
                                        transferred = True
                                    except Exception as e:
                                        print(f"[Projection] Error projecting {student_key}: {e}")
                except (ValueError, IndexError) as e:
                    print(f"[Projection] Error parsing block index in {student_key}: {e}")
        
        # If we got here and nothing was transferred, mark as skipped
        if not transferred:
            projection_stats['skipped'] += 1
    
    # Step 4: Report projection statistics
    print()
    print(f"[Projection] ================================================================================")
    print(f"[Projection] Projection Complete - Summary:")
    print(f"[Projection] ================================================================================")
    print(f"  ✓ Exact matches:          {projection_stats['exact_matches']:4d} weights")
    print(f"  ✓ Projected (dim change): {projection_stats['projected']:4d} weights")
    print(f"  ✓ Conv3D → Conv2D:        {projection_stats['conv3d_to_2d']:4d} conversions")
    print(f"  ⚠ Skipped (random init):  {projection_stats['skipped']:4d} weights")
    
    total_weights = sum(projection_stats.values())
    if total_weights > 0:
        transfer_rate = (projection_stats['exact_matches'] + projection_stats['projected']) / total_weights * 100
        print(f"  → Transfer rate:          {transfer_rate:.1f}%")
    print(f"[Projection] ================================================================================")
    print()

    student_model.to(device)
    return student_model