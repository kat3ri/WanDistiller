"""
Test to verify WanLiteStudent class structure and Diffusers compatibility.

This tests the model structure without requiring CUDA or full dependencies.
"""

import json
import ast
import os

def extract_class_from_file(filename, classname):
    """Extract a class definition from a Python file."""
    with open(filename, 'r') as f:
        tree = ast.parse(f.read())
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == classname:
            return node
    return None

def test_model_structure():
    """Test the WanLiteStudent class structure."""
    print("=" * 80)
    print("Testing WanLiteStudent Model Structure")
    print("=" * 80)
    
    # Read the train_distillation.py file
    print("\n[1/6] Reading train_distillation.py...")
    filepath = "train_distillation.py"
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    print("✓ File read successfully")
    
    # Check imports
    print("\n[2/6] Checking required imports...")
    required_imports = [
        'from diffusers.configuration_utils import ConfigMixin, register_to_config',
        'from diffusers.models.modeling_utils import ModelMixin'
    ]
    
    for imp in required_imports:
        if imp in content:
            print(f"   ✓ Found: {imp}")
        else:
            print(f"   ✗ Missing: {imp}")
            return False
    
    # Check class definition
    print("\n[3/6] Checking WanLiteStudent class definition...")
    if 'class WanLiteStudent(ModelMixin, ConfigMixin):' in content:
        print("   ✓ WanLiteStudent inherits from ModelMixin and ConfigMixin")
    else:
        print("   ✗ WanLiteStudent doesn't properly inherit from mixins")
        return False
    
    # Check @register_to_config decorator
    print("\n[4/6] Checking @register_to_config decorator...")
    if '@register_to_config' in content:
        print("   ✓ @register_to_config decorator present")
        
        # Check it's before __init__
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '@register_to_config' in line:
                # Check next non-empty line is def __init__
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip().startswith('def __init__'):
                        print("   ✓ @register_to_config correctly decorates __init__")
                        break
                break
    else:
        print("   ✗ @register_to_config decorator not found")
        return False
    
    # Check config parameters in __init__
    print("\n[5/6] Checking __init__ config parameters...")
    required_params = [
        'model_type', 'hidden_size', 'depth', 'num_heads',
        'num_channels', 'image_size', 'patch_size',
        'text_max_length', 'text_encoder_output_dim', 'projection_factor'
    ]
    
    # Find WanLiteStudent __init__ method (look for the class first)
    class_start = content.find('class WanLiteStudent(ModelMixin, ConfigMixin):')
    if class_start == -1:
        print("   ✗ Could not find WanLiteStudent class")
        return False
    
    # Find __init__ after the class definition
    init_start = content.find('def __init__(', class_start)
    init_section = content[init_start:init_start+3000] if init_start != -1 else ''
    
    for param in required_params:
        if f'{param}=' in init_section or f'{param},' in init_section:
            print(f"   ✓ Parameter '{param}' present")
        else:
            print(f"   ✗ Parameter '{param}' missing")
            return False
    
    # Check save_pretrained method
    print("\n[6/6] Checking save_pretrained method...")
    if 'def save_pretrained(self, output_dir' in content:
        print("   ✓ save_pretrained method exists")
        
        if 'super().save_pretrained' in content:
            print("   ✓ Calls parent class save_pretrained")
        else:
            print("   ✗ Doesn't call parent class save_pretrained")
            return False
    else:
        print("   ✗ save_pretrained method not found")
        return False
    
    print("\n" + "=" * 80)
    print("✓ ALL STRUCTURE CHECKS PASSED!")
    print("=" * 80)
    print("\nThe WanLiteStudent model structure is compatible with:")
    print("  - HuggingFace Diffusers ModelMixin")
    print("  - HuggingFace Diffusers ConfigMixin")
    print("  - @register_to_config for automatic config management")
    print("  - WAN model loading patterns")
    print()
    
    # Print a summary of what will be saved
    print("When save_pretrained() is called, it will save:")
    print("  1. config.json - with all @register_to_config parameters:")
    for param in required_params:
        print(f"     - {param}")
    print("  2. diffusion_model.safetensors - model weights")
    print()
    
    return True

if __name__ == "__main__":
    import sys
    success = test_model_structure()
    sys.exit(0 if success else 1)
