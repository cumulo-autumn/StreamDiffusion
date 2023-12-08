import glob
import os
from typing import *

import fire
import PIL.Image
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion


def main(
    input: str, output: str, prompt: str = "Girl with panda ears wearing a hood", width: int = 512, height: int = 512
):
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

    images = glob.glob(os.path.join(input, "*"))
    images = images + [images[-1]] * (stream.batch_size - 1)
    outputs = []

    for i in range(stream.batch_size - 1):
        image = images.pop(0)
        outputs.append(image)
        input_image = PIL.Image.open(image).convert("RGB")
        output_x = stream(input_image)
        output_image = stream.image_processor.postprocess(output_x, output_type="pil")[0]
        output_image.save(os.path.join(output, f"{i}.png"))

    for image in images:
        outputs.append(image)
        try:
            input_image = PIL.Image.open(image).convert("RGB")
        except Exception:
            continue

        output_x = stream(input_image)
        output_image = stream.image_processor.postprocess(output_x, output_type="pil")[0]

        name = outputs.pop(0)
        basename = os.path.basename(name)
        output_image.save(os.path.join(output, basename))


if __name__ == "__main__":
    fire.Fire(main)
