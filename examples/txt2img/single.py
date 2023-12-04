from typing import *

import fire
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image


def main(output: str, prompt: str = "Girl with panda ears wearing a hood", width: int = 512, height: int = 512):
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file("./model.safetensors").to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    pipe.enable_xformers_memory_efficient_attention()

    stream = StreamDiffusion(
        pipe, [0, 16, 32, 45], torch_dtype=torch.float16, width=width, height=height, is_drawing=True
    )
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
    stream.load_lcm_lora()
    stream.fuse_lora()
    stream.prepare(
        prompt,
        num_inference_steps=50,
    )

    for _ in range(stream.batch_size - 1):
        stream.txt2img()

    output_x = stream.txt2img()
    output_image = postprocess_image(output_x, output_type="pil")[0]
    output_image.save(output)


if __name__ == "__main__":
    fire.Fire(main)
