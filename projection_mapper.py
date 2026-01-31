import torch
import torch.nn as nn
from .student_config import StudentConfig


def load_and_project_weights(student_model, teacher_state_dict, device="cuda"):
    """
    Loads weights from a Teacher state_dict into the Student model.
    Handles dimension mismatches by injecting Projection layers.
    """
    student_state_dict = student_model.state_dict()

    print(f"[Projection] Initializing Student with Teacher weights...")

    # Mapping of student layer names to teacher layer names
    # Assuming Teacher is Wan 2.2 (approx dimensions)
    layer_mappings = {
        "conv_in.weight": "conv_in.weight",
        "conv_in.bias": "conv_in.bias",

        "text_proj.weight": "text_proj.weight",
        "text_proj.bias": "text_proj.bias",

        # Attention projections (Mapping from Teacher dim to Student dim)
        # Example: Teacher Linear(1152, 1152) -> Student Linear(640, 640)
        # We need to inject a new layer in the student model to handle the downprojection
    }

    # Iterate over student model state dict to find keys
    for student_key, student_param in student_state_dict.items():

        # 1. Exact Match Case
        if student_key in teacher_state_dict:
            teacher_param = teacher_state_dict[student_key]

            if param.shape == teacher_param.shape:
                student_param.data.copy_(teacher_param.data)
                continue

            # 2. Dimension Mismatch Case (Need Projection)
            print(f"[Projection] Mismatch detected: {student_key}")
            print(f"    Student Shape: {student_param.shape}")
            print(f"    Teacher Shape: {teacher_param.shape}")

            # Create a projection layer
            # We determine the input and output dims based on the tensors
            in_dim = teacher_param.shape[-1]
            out_dim = student_param.shape[-1]

            # Determine layer type (Linear or Conv2d)
            if "conv_in" in student_key:
                proj_layer = nn.Conv2d(4, out_dim, kernel_size=3, padding=1)
            elif "text_proj" in student_key:
                proj_layer = nn.Linear(in_dim, out_dim)
            else:
                # Default to Linear for block internal projections
                proj_layer = nn.Linear(in_dim, out_dim)

            # Initialize projection layer with teacher weights
            # For Conv2d, we permute/flatten weights to fit Linear input
            if len(teacher_param.shape) == 4:
                proj_layer.weight.data.copy_(teacher_param.permute(3, 2, 0, 1).flatten())
            else:
                proj_layer.weight.data.copy_(teacher_param.data)

            # Copy bias
            if teacher_param.shape[0] == student_param.shape[-1]:
                proj_layer.bias.data.copy_(teacher_param.data)

            # Scale weights for stability
            scale = (in_dim / out_dim) ** 0.5
            proj_layer.weight.data *= scale

            # Inject this projection layer into the student model
            # We modify the specific module directly to ensure it's in the hierarchy
            # Note: This assumes the module structure defined in your Student class
            if "conv_in" in student_key:
                student_model.conv_in = proj_layer
            elif "text_proj" in student_key:
                student_model.text_proj = proj_layer
            else:
                # For blocks, we inject into the attention modules
                # This part requires knowing the block index, simplified here:
                # Ideally, you map 'blocks.0.attn.proj.weight' -> 'teacher.blocks.0.attn.proj.weight'
                # and replace that specific module's weight with the projection layer
                pass

            # Ideally, you would update the specific submodule (e.g., student.blocks[0].attn.proj)
            # with the new projection layer. 
            # For this example, we rely on the student class to use these weights or 
            # modify the forward pass to use the injected layer.

            # Update the local tensor reference for the final save
            # (This is a simplified approach; in practice, you might re-init the whole module)
            student_param.data = proj_layer.weight  # Placeholder for saving

        else:
            # Random initialization for missing keys (usually not ideal for distillation, but safe fallback)
            nn.init.zeros_(student_param)
            print(f"[Projection] Missing key in teacher: {student_key} -> Zeroing out.")

    student_model.to(device)
    print("[Projection] Initialization complete.")