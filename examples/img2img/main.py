import os
from typing import *

import fire
import PIL.Image
import torch
from diffusers import AutoencoderTiny, LCMScheduler, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import pil2tensor, postprocess_image


def main(input: str, output: str, scale: int = 1):
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file("./model.safetensors").to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()
    pipe.enable_xformers_memory_efficient_attention()

    input_image = PIL.Image.open(os.path.join(input))
    width = int(input_image.width * scale)
    height = int(input_image.height * scale)

    stream = StreamDiffusion(
        pipe,
        [35, 45],
        torch_dtype=torch.float16,
        width=width,
        height=height,
    )
    stream.prepare(
        "Girl with panda ears wearing a hood",
        num_inference_steps=50,
        generator=torch.manual_seed(2),
    )

    input_image = input_image.resize((width, height))
    input_tensor = pil2tensor(input_image)

    for _ in range(stream.batch_size - 1):
        stream(input_tensor.detach().clone().to(device=stream.device, dtype=stream.dtype))

    output_x = stream(input_tensor.detach().clone().to(device=stream.device, dtype=stream.dtype))
    output_image = postprocess_image(output_x, output_type="pil")[0]
    output_image.save(output)


if __name__ == "__main__":
    fire.Fire(main)
