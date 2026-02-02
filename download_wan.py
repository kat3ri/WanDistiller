
import torch
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

# 1. Define device and data type
# Use "cuda" for GPU acceleration. Use bfloat16 for potential memory savings if supported.
dtype = torch.bfloat16
device = "cuda"

# 2. Load the VAE and the main pipeline
# The 'Wan-AI/Wan2.2-T2V-A14B-Diffusers' model is integrated into the diffusers library.

model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
pipe.to(device)

# 3. (Optional) Adjust the flow_shift parameter for different resolutions
# 5.0 for 720P, 3.0 for 480P.
flow_shift = 5.0
# The scheduler might need to be configured depending on the version
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift) 


# 4. Define your prompt and generation parameters
prompt = "Two anthropomorphic cats in comfy boxing gear fight on a spotlighted stage."
height = 720
width = 1280
num_frames = 1
num_inference_steps = 40
guidance_scale = 4.0

# 5. Run the pipeline to force the model to cache
# The result is a list of PIL.Image objects.
output = pipe(
    prompt=prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
)
frames = output.frames[0]

# 6. Export the frames to a video file (requires an appropriate video codec like 'mp4v')
video_path = "wan2_2_output.mp4"
#export_to_video(frames, video_path, fps=24) # 24 fps is a common standard

print(f"Wan Downloaded and cached")
