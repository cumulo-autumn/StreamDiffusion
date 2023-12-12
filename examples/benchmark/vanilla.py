import io

import fire
import PIL.Image
import requests
import torch
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline, DDPMScheduler
from diffusers.utils import load_image

def download_image(url: str):
    response = requests.get(url)
    image = PIL.Image.open(io.BytesIO(response.content))
    return image

def run(
    warmup: int = 10,
    iterations: int = 100,
    model_id: str = "stabilityai/sd-turbo",
    prompt: str = "Girl with panda ears wearing a hood",
    width: int = 512,
    height: int = 512,
): 
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()
    pipe.to("cuda")
    scheduler_config = {
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "clip_sample": False,
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
        "sample_max_value": 1.0,
        "steps_offset": 1,
        "timestep_spacing": "trailing",
        "trained_betas": None,
    }
    scheduler = DDPMScheduler(**scheduler_config)
    pipe.scheduler = scheduler

    init_image = load_image("https://github.com/ddpn08.png").convert("RGB").resize((width, height))

    for _ in range(warmup):
        pipe(prompt=prompt, image=init_image, timesteps=[99, 30], guidance_scale=0.0)
    
    results = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in tqdm(range(iterations)):
        start.record()
        pipe(prompt=prompt, image=init_image, timesteps=[99, 30], guidance_scale=0.0)
        end.record()

        torch.cuda.synchronize()
        results.append(start.elapsed_time(end))

    print(f"Average time: {sum(results) / len(results)} ms")
    print(f"Average FPS: {1000 / (sum(results) / len(results))}")
    import numpy as np
    fps_arr = 1000/np.array(results)
    print(f"Max FPS: {np.max(fps_arr)}")
    print(f"Min FPS: {np.min(fps_arr)}")
    print(f"Std: {np.std(fps_arr)}")

if __name__ == "__main__":
    fire.Fire(run)