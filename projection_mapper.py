import torch
import torch.nn as nn
import sys
import json
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


def load_and_project_weights(student_model, teacher_state_dict, config=None, device="cuda"):
    """
    Loads weights from Teacher (Wan 2.2) to Student.
    Robustly handles dimension mismatches by injecting projection layers.
    """
    if config is None:
        config = StudentConfig()

    student_state_dict = student_model.state_dict()

    print(f"[Projection] Initializing Student from Teacher weights...")
    print(f"[Projection] Config Hidden Size: {config.hidden_size}")

    # Iterate over student parameters to find keys to match
    for student_key, student_param in student_state_dict.items():

        # --- 1. Exact Match ---
        if student_key in teacher_state_dict:
            teacher_param = teacher_state_dict[student_key]

            # If shapes match, simply copy weights
            if student_param.shape == teacher_param.shape:
                student_param.data.copy_(teacher_param.data)
                continue

            # --- 2. Dimension Mismatch (Projection Required) ---
            print(f"[Projection] Mismatch detected: {student_key}")
            print(f"    Student Shape: {student_param.shape}")
            print(f"    Teacher Shape: {teacher_param.shape}")

            # Logic to determine input/output dimensions
            in_dim = teacher_param.shape[-1]
            out_dim = student_param.shape[-1]

            # Determine layer type
            if "conv_in" in student_key:
                proj_layer = nn.Conv2d(4, out_dim, kernel_size=3, padding=1, bias=False)
            elif "text_proj" in student_key:
                proj_layer = nn.Linear(in_dim, out_dim, bias=False)
            elif "time_embed" in student_key:
                # Time embed is often a Linear layer
                proj_layer = nn.Linear(in_dim, out_dim, bias=False)
            elif "blocks" in student_key:
                # Handle Block projections (e.g., Attention weights)
                # This assumes we are down-projecting (Teacher is larger)
                # We create a Linear layer to project the features
                proj_layer = nn.Linear(in_dim, out_dim, bias=False)
            else:
                print(f"[Projection] Unknown layer type for {student_key}, skipping projection.")
                continue

            # --- 3. Weight Initialization ---

            # Copy teacher weights (handle Permute for Conv2d)
            if len(teacher_param.shape) == 4:
                # Conv2d weights: (out_channels, in_channels, kH, kW)
                # Teacher often has (in_channels, out_channels, kH, kW)
                # We reshape to match Conv2d expectations
                proj_layer.weight.data.copy_(teacher_param.permute(3, 2, 0, 1).flatten())
            else:
                proj_layer.weight.data.copy_(teacher_param.data)

            # --- 4. Scale Weights ---
            # Standard He initialization scaling for projection
            scale = (in_dim / out_dim) ** 0.5
            proj_layer.weight.data *= scale

            # --- 5. Inject into Student Model ---

            # We must replace the specific module in the student hierarchy
            # The key format determines where we inject

            # Example: student.blocks.0.attn.proj
            parts = student_key.split('.')

            if len(parts) >= 2 and parts[0] == 'blocks':
                # Access the specific block and submodule
                block_idx = int(parts[1])
                submodule_name = '.'.join(parts[2:])

                if block_idx < len(student_model.blocks):
                    target_module = student_model.blocks[block_idx]

                    # Traverse to the final submodule (e.g., 'attn.proj')
                    current_mod = target_module
                    for name in parts[2:-1]:
                        if hasattr(current_mod, name):
                            current_mod = getattr(current_mod, name)

                    # Replace the final submodule (e.g., 'proj') with the new projection layer
                    if parts[-1] in current_mod:
                        print(f"[Projection] Injecting projection layer at {student_key}")
                        current_mod._modules[parts[-1]] = proj_layer
                        # Update the reference in the block's module registry if necessary
                        # (In some custom classes, this might require adding the layer directly)

            # Specific overrides for top-level layers
            elif "conv_in" in student_key:
                student_model.conv_in = proj_layer
            elif "text_proj" in student_key:
                student_model.text_proj = proj_layer
            elif "time_embed" in student_key:
                student_model.time_embed = proj_layer

    student_model.to(device)
    print("[Projection] Initialization complete.")