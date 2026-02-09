"""
Test to verify distributed training step count calculations are correct.
"""

def test_step_calculations():
    """Test the step calculation logic for distributed training."""
    
    # Test case 1: 2000 prompts, 8 GPUs, batch_size=1
    dataset_size = 2000
    world_size = 8
    batch_size = 1
    
    # Calculate prompts per GPU (with padding)
    prompts_per_gpu = dataset_size // world_size
    if dataset_size % world_size != 0:
        prompts_per_gpu += 1
    
    # Calculate steps per GPU
    steps_per_epoch_per_gpu = (prompts_per_gpu + batch_size - 1) // batch_size
    
    # Total steps across all GPUs
    total_steps_per_epoch = steps_per_epoch_per_gpu * world_size
    
    print(f"Test Case 1: {dataset_size} prompts, {world_size} GPUs, batch_size={batch_size}")
    print(f"  Prompts per GPU: {prompts_per_gpu}")
    print(f"  Steps per GPU per epoch: {steps_per_epoch_per_gpu}")
    print(f"  Total steps across all GPUs: {total_steps_per_epoch}")
    
    # With 2000 prompts and 8 GPUs, each GPU should get 250 prompts
    assert prompts_per_gpu == 250, f"Expected 250, got {prompts_per_gpu}"
    assert steps_per_epoch_per_gpu == 250, f"Expected 250, got {steps_per_epoch_per_gpu}"
    assert total_steps_per_epoch == 2000, f"Expected 2000, got {total_steps_per_epoch}"
    
    print("✓ Test Case 1 passed!")
    
    # Test case 2: 2001 prompts, 8 GPUs, batch_size=1 (not evenly divisible)
    dataset_size = 2001
    world_size = 8
    batch_size = 1
    
    prompts_per_gpu = dataset_size // world_size
    if dataset_size % world_size != 0:
        prompts_per_gpu += 1
    
    steps_per_epoch_per_gpu = (prompts_per_gpu + batch_size - 1) // batch_size
    total_steps_per_epoch = steps_per_epoch_per_gpu * world_size
    
    print(f"\nTest Case 2: {dataset_size} prompts, {world_size} GPUs, batch_size={batch_size}")
    print(f"  Prompts per GPU: {prompts_per_gpu}")
    print(f"  Steps per GPU per epoch: {steps_per_epoch_per_gpu}")
    print(f"  Total steps across all GPUs: {total_steps_per_epoch}")
    
    # With 2001 prompts and 8 GPUs, some GPUs get 251, some get 250
    # The calculation gives 251 to all (accounting for padding)
    assert prompts_per_gpu == 251, f"Expected 251, got {prompts_per_gpu}"
    assert steps_per_epoch_per_gpu == 251, f"Expected 251, got {steps_per_epoch_per_gpu}"
    # Total will be 251 * 8 = 2008 (due to padding)
    assert total_steps_per_epoch == 2008, f"Expected 2008, got {total_steps_per_epoch}"
    
    print("✓ Test Case 2 passed!")
    
    # Test case 3: 2000 prompts, 4 GPUs, batch_size=2
    dataset_size = 2000
    world_size = 4
    batch_size = 2
    
    prompts_per_gpu = dataset_size // world_size
    if dataset_size % world_size != 0:
        prompts_per_gpu += 1
    
    steps_per_epoch_per_gpu = (prompts_per_gpu + batch_size - 1) // batch_size
    total_steps_per_epoch = steps_per_epoch_per_gpu * world_size
    
    print(f"\nTest Case 3: {dataset_size} prompts, {world_size} GPUs, batch_size={batch_size}")
    print(f"  Prompts per GPU: {prompts_per_gpu}")
    print(f"  Steps per GPU per epoch: {steps_per_epoch_per_gpu}")
    print(f"  Total steps across all GPUs: {total_steps_per_epoch}")
    
    # With 2000 prompts and 4 GPUs, each GPU gets 500 prompts
    # With batch_size=2, each GPU has 250 steps
    assert prompts_per_gpu == 500, f"Expected 500, got {prompts_per_gpu}"
    assert steps_per_epoch_per_gpu == 250, f"Expected 250, got {steps_per_epoch_per_gpu}"
    assert total_steps_per_epoch == 1000, f"Expected 1000, got {total_steps_per_epoch}"
    
    print("✓ Test Case 3 passed!")
    
    # Test case 4: Single GPU (non-distributed)
    dataset_size = 2000
    batch_size = 1
    
    steps_per_epoch = (dataset_size + batch_size - 1) // batch_size
    
    print(f"\nTest Case 4: {dataset_size} prompts, single GPU, batch_size={batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    
    assert steps_per_epoch == 2000, f"Expected 2000, got {steps_per_epoch}"
    
    print("✓ Test Case 4 passed!")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_step_calculations()
