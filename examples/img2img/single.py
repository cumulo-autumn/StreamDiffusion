import os
from typing import *

import fire
import PIL.Image
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import pil2tensor, postprocess_image


def main(input: str, output: str, prompt: str = "Girl with panda ears wearing a hood", scale: int = 1):
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file("../../model.safetensors").to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    pipe.enable_xformers_memory_efficient_attention()

    stream = StreamDiffusion(
        pipe, [32, 40, 45], torch_dtype=torch.float16, width=width, height=height, is_drawing=True
    )
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
    stream.load_lcm_lora()
    stream.fuse_lora()
    stream.prepare(
        prompt,
        num_inference_steps=50,
    )

    input_image = PIL.Image.open(os.path.join(input))
    width = int(input_image.width * scale)
    height = int(input_image.height * scale)

    stream = StreamDiffusion(
        pipe,
        [22, 32, 45],
        torch_dtype=torch.float16,
        width=width,
        height=height,
    )
    stream.prepare(
        "beach, sea, a palm tree, clouds",
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
