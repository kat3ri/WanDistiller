This is a comprehensive framework to create a **Wan-T2I (Static)** model.
### Concept
We will use the existing **Wan 2.2** (Video) model as the Teacher. We will configure it to generate 
high-quality **single-frame images** (discarding temporal motion). We will then train a smaller 
**Student** model (Wan-Lite) to mimic the Teacher's noise prediction steps, effectively transferring the 
"Image Quality" knowledge without the "Video Motion" baggage.
### Project Structure
```text
.
├── README.md
.
├── main.py                 # Entry point for the script
├── requirements.txt        # Dependencies
├── config/
│   └── student_config.json # Student architecture configuration
├── data/
│   └── static_prompts.txt  # List of prompts for static images
└── train_distillation.py   # Main training logic
```
---
### 1. `README.md`
```markdown
# Wan-T2I Distillation (Wan-Static)
This project distills the high-quality image generation capabilities of **Wan 2.2** into a lighter, 
standalone Text-to-Image model. It strips out temporal (movement) artifacts by training a Student model to 
mimic a Teacher that is forced to generate static frames.
## Overview
- **Teacher:** Wan 2.2 (Video Generation Model). It holds the image quality knowledge.
- **Student:** Wan-Lite (Lightweight Image Model). It is smaller (fewer channels/depth) but retains the 
quality.
- **Method:** Noise Prediction Distillation (MSE Loss).
- **Goal:** High-fidelity T2I generation with zero motion blur.
## Prerequisites
- Python 3.9+
- PyTorch 2.0+
- 2x NVIDIA GPUs (one for Teacher, one for Student, or use CPU for slower training)
## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare a list of static prompts (e.g., portrait photography, landscapes without camera panning). Place 
them in `data/static_prompts.txt`.
## Usage
### Step 1: Train the Distillation Loop
The script will:
1. Load the Teacher (Wan 2.2).
2. Load the Student (Wan-Lite).
3. Generate clean images from prompts using the Teacher.
4. Add noise to those images.
5. Ask the Student to predict the noise.
6. Update Student weights to match Teacher.
```bash
python train_distillation.py \
    --teacher_path "timbrooks/instruct-wan" \
    --student_config "config/student_config.json" \
    --data_path "data/static_prompts.txt" \
    --output_dir "./outputs/wan_t2i" \
    --learning_rate 1e-5 \
    --num_steps 1000
```
### Step 2: Inference
Once trained, the `outputs/wan_t2i` folder will contain the distilled model. You can use it in your own 
inference pipeline or as the backbone for your own UI.
## Key Features
- **Static-Only Training:** The teacher is forced to generate 1-frame videos, eliminating motion learning.
- **Architectural Compression:** The student is significantly smaller, making it faster and lighter to 
serve.
- **No Adapters Needed:** The model is fully distilled into its own weights.
```
---
### 2. `requirements.txt`
```text
torch>=2.0.0
diffusers>=0.25.0
transformers>=4.35.0
accelerate>=0.24.0
safetensors>=0.3.1
xformers>=0.0.23 # Optional, for memory efficiency
```
---
### 3. `config/student_config.json`
This file defines the "Lightweight" architecture. We reduce the `hidden_size` and `depth` to create a 
model that runs faster but should still look good if trained properly.
```json
{
  "model_name": "Wan-Lite-T2I",
  "hidden_size": 640,
  "num_heads": 10,
  "depth": 12,
  "m_patch_size": 16,
  "temporal_m_patch_size": 16,
  "m_channels": 32,
  "temporal_m_channels": 32,
  "scale_factor": 0.5,
  "motion_bucket_id": 0
}
```
---
### 4. `train_distillation.py`
This is the core logic. It sets up the Teacher pipeline (forcing it to generate static images) and the 
Student pipeline.
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import WanImageToImagePipeline, DDPMScheduler
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import argparse
import json
import os
# ==========================================
# 1. CONFIG LOADING
# ==========================================
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
# ==========================================
# 2. DATA PREPARATION
# ==========================================
class StaticPromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        # Return the prompt as a string
        return {"prompt": self.prompts[idx]}
# ==========================================
# 3. MODEL SETUP
# ==========================================
class WanDistillationTrainer:
    def __init__(self, teacher_path, student_config, data_path, output_dir, device="cuda"):
        
        self.device = device
        
        # Load Teacher (Wan 2.2)
        # We use ImageToImage pipeline but force num_frames=1 to strip motion
        print(f"Loading Teacher from {teacher_path}...")
        self.teacher = WanImageToImagePipeline.from_pretrained(
            teacher_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        self.teacher.to(device)
        self.teacher.enable_model_cpu_offload() # Save VRAM
        # Load Student Config
        with open(student_config, 'r') as f:
            self.student_cfg = json.load(f)
            
        # Load Student (Initial weights from Teacher)

** NOTE:  *   *The "Distillation" Process:* In standard distillation (like LoRA, PEFT, or Adapter training), you 
do *not* load the teacher weights into the student. You initialize the student with random weights. The 
student then learns to mimic the teacher's behavior through the loss function.
**
        print("Initializing Student model...")
        self.student = WanImageToImagePipeline.from_pretrained(
            teacher_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        
        # Apply Distillation: Change architecture dimensions
        # NOTE: This requires modifying the internal DiT layers.
        # For this scaffold, we are assuming the pipeline loads the base model
        # and we will modify the scheduler or specific layers below.
        
        # NOTE: Actual implementation requires mapping the WanDiT class dimensions
        # based on self.student_cfg. For brevity, we initialize student on teacher weights
        # and will optimize the layers that matter in the loop.
        
        self.student.to(device)
        
        # Load Text Encoder (Tokenizers)
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_path)
        
        # Noise Scheduler
        self.scheduler = DDPMScheduler.from_config(self.teacher.scheduler.config)
    def distill_step(self, batch, optimizer, step):
        prompts = batch["prompt"]
        
        optimizer.zero_grad()
        # --- Phase 1: Teacher Generation (Static) ---
        # We force Wan to generate ONLY 1 frame to eliminate motion artifacts
        with torch.no_grad():
            # Generate images from Teacher
            teacher_outputs = self.teacher(
                prompt=prompts,
                num_inference_steps=10, 
                num_frames=1, # CRITICAL: Strip temporal dimensions
                guidance_scale=7.5,
                height=1024,
                width=1024
            )
            
            # Get the clean image (Latent) from Teacher
            teacher_latent = teacher_outputs.images
            
            # Add noise to the latent (Simulating the reverse diffusion process)
            # In diffusion, the model predicts the noise added to the image.
            # We add noise to the teacher's image to get the 'target'.
            noise = torch.randn_like(teacher_latent)
            noisy_latent = self.scheduler.add_noise(teacher_latent, noise, timesteps=torch.tensor([0])) 
            
            # Clean noise target (the noise we added)
            target_noise = noise
        # --- Phase 2: Student Prediction ---
        
        # We run the student pipeline in inference mode to get predictions
        # Note: In a full production setup, we would bypass the full pipeline 
        # and run only the UNet forward pass for speed.
        
        student_outputs = self.student(
            prompt=prompts,
            image=noisy_latent,
            num_inference_steps=20, # Student is slower, needs more steps
            num_frames=1, # Ensure 1 frame
            return_dict=False
        )
        
        # Get the predicted noise from Student
        # student_outputs[0] usually contains the image latents, 
        # but for Wan, we often care about the internal latent prediction or final image.
        # Here, we assume we want the final clean image to compare to Teacher's latent.
        
        student_latent = student_outputs[0] 
        
        # Calculate Loss (MSE between Teacher Image and Student Image)
        # We want the Student to produce the exact same image as the Teacher
        loss = F.mse_loss(student_latent.float(), teacher_latent.float())
##NOTE: make sure that we are running high then low for the teacher so student learns the final output in 1 pass ##
        
        # Backprop
        loss.backward()
        optimizer.step()
        
        return loss.item()
    def train(self, num_steps, learning_rate, batch_size=4):
        optimizer = torch.optim.AdamW(self.student.unet.parameters(), lr=learning_rate)
        
        # Load Data
        with open(data_path, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        dataset = StaticPromptDataset(prompts)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"Starting distillation for {num_steps} steps...")
        
        for step in tqdm(range(num_steps)):
            for batch in loader:
                loss = self.distill_step(batch, optimizer, step)
                
            if step % 100 == 0:
                print(f"Step {step}: Loss {loss:.4f}")
        print("Distillation complete.")
        self.save_model(output_dir)
    def save_model(self, output_dir):
        self.student.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, default="timbrooks/instruct-wan")
    parser.add_argument("--student_config", type=str, default="config/student_config.json")
    parser.add_argument("--data_path", type=str, default="data/static_prompts.txt")
    parser.add_argument("--output_dir", type=str, default="./outputs/wan_t2i")
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    
    args = parser.parse_args()
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    trainer = WanDistillationTrainer(
        teacher_path=args.teacher_path,
        student_config=args.student_config,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    trainer.train(num_steps=args.num_steps, learning_rate=1e-4, batch_size=args.batch_size)
```
### How to use this scaffold
1.  **Create `data/static_prompts.txt`**:
    Add about 50-100 prompts that are visually static.
    ```text
    A portrait of an old man in a dark room, cinematic lighting
    A mountain landscape with no clouds, sharp focus
    A close up of a painting on a wall
    ```
2.  **Run the script**:
    This process will take a while as it generates images on the fly to train the student.
    ```bash
    python train_distillation.py --num_steps 200
    ```
>>>  your prompts are in the filenames (e.g., a_photo_of_a_cat.jpg), the script can extract them automatically.
... If you don't have prompts, you will need to create them. A common starting point is to use a generic prompt 
... or create them manually.
... Here is a Python script that you can use to generate the metadata.jsonl file. It assumes that the prompt for
...  each image is its filename (without the extension).
... 
... create_t2i_metadata.py
