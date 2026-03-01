# No borrar estos comentarios:
# Características del PC de pruebas:
# NVIDIA GeForce 1060 de 6GB, 96GB de RAM, CPU Intel Xeon E5-2650 v4
# Híbrido de SDXL + AnimateDiff + Upscale frame to frame

import torch
import os
from diffusers import StableDiffusionXLPipeline
from diffusers import AnimateDiffPipeline, MotionAdapter
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
from diffusers.utils import export_to_video

device = "cuda"
base_prompt="ultra realistic cinematic portrait of a woman, natural skin texture, full body, shallow depth of field, symmetrical eyes, detailed iris, natural eye color"
negative_prompt = "deformed, plastic skin, low quality, distorted face, bad anatomy, extra fingers, extra limbs, deformed face, asymmetrical eyes, warped mouth, blue artifacts, chromatic aberration, color distortion, iris deformation"

keyframe_path = "keyframe.png"
if not os.path.exists(keyframe_path):
    print("Generating keyframe for SDXL...\n")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    )#.to(device)

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    #pipe.enable_xformers_memory_efficient_attention()

    generator = torch.Generator(device=device).manual_seed(42)

    image = pipe(
        prompt = base_prompt,
        negative_prompt = negative_prompt,
        num_inference_steps = 24,
        guidance_scale = 6.0,
        generator = generator,
        height = 768,
        width = 768
    ).images[0]

    image.save("keyframe.png")

    del pipe
    torch.cuda.empty_cache()
else:
    print("Keyframe exists. Generating SDXL...\n")

# reset params and start with AnimateDiff
device = "cuda"

motion_adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5",
    torch_dtype=torch.float16
)

pipe = AnimateDiffPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    motion_adapter=motion_adapter,
    torch_dtype=torch.float16
).to(device)

#pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    #num_train_timesteps=1000,
    #beta_start=0.00085,
    #beta_end=0.012,
    #beta_schedule="scaled_linear",
    pipe.scheduler.config,
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True
)

#pipe.enable_attention_slicing() #activo antes
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()
#pipe.enable_sequential_cpu_offload()
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
#pipe.enable_xformers_memory_efficient_attention()

init_image = Image.open("keyframe.png").resize((512, 512))

generator = torch.Generator(device=device).manual_seed(42)

video = pipe(
    prompt = base_prompt +", subtle breathing, blinking, cinematic lighting, realistic movement",
    negative_prompt = negative_prompt,
    #image = init_image,
    #strength = 0.42,
    num_frames = 8,# 12 para 9min
    num_inference_steps = 12, #16 para 9min
    guidance_scale = 5.0, #6.0 antes
    generator = generator,    
    height = 512,
    width = 512
).frames[0]

export_to_video(video, "hibrid_output.mp4", fps=12)

print("Video generado\n")