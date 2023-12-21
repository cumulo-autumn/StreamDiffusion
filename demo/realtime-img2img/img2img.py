from diffusers import (
    AutoPipelineForImage2Image,
    AutoencoderTiny,
)
from wrapper import StreamDiffusionWrapper

import torch

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math

base_model = "stabilityai/sd-turbo"
taesd_model = "madebyollin/taesd"

default_prompt = "Portrait a disney character cinematic, masterpiece"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

page_content = """
<h1 class="text-3xl font-bold">Real-Time Latent Consistency Model</h1>
<h3 class="text-xl font-bold">Image-to-Image LCM</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://huggingface.co/blog/lcm_lora"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">LCM</a>
Image to Image pipeline using
    <a
    href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/lcm#performing-inference-with-lcm"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">Diffusers</a
    > with a MJPEG stream server.
</p>
<p class="text-sm text-gray-500">
    Change the prompt to generate different images, accepts <a
    href="https://github.com/damian0815/compel/blob/main/doc/syntax.md"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">Compel</a
    > syntax.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "img2img"
        title: str = "Image-to-Image LCM"
        description: str = "Generates an image from a text prompt"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        negative_prompt: str = Field(
            default_negative_prompt,
            title="Negative Prompt",
            field="textarea",
            id="negative_prompt",
        )
        seed: int = Field(
            2159232, min=0, title="Seed", field="seed", hide=True, id="seed"
        )
        steps: int = Field(
            4, min=1, max=15, title="Steps", field="range", hide=True, id="steps"
        )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        guidance_scale: float = Field(
            1.2,
            min=0,
            max=20,
            step=0.001,
            title="Guidance Scale",
            field="range",
            hide=True,
            id="guidance_scale",
        )
        strength: float = Field(
            0.5,
            min=0.25,
            max=1.0,
            step=0.001,
            title="Strength",
            field="range",
            hide=True,
            id="strength",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        params = self.InputParams()
        print("PARAMS:", params)
        # self.stream = StreamDiffusionWrapper(
        #     model_id=base_model,
        #     t_index_list=[32, 40, 45],
        #     frame_buffer_size=1,
        #     width=params.width,
        #     height=params.height,
        #     warmup=10,
        #     acceleration=args.acceleration,
        #     is_drawing=True,
        #     mode="img2img",
        #     use_denoising_batch=args.use_denoising_batch,
        #     cfg_type=args.cfg_type,
        #     output_type="pil",
        #     lcm_lora_id=None,
        #     vae_id=None,
        #     device=device,
        #     dtype=torch_dtype,
        #     use_lcm_lora=True,
        #     use_tiny_vae=args.taesd,
        # )
        self.stream = StreamDiffusionWrapper(
            model_id=base_model,
            t_index_list=[32, 40, 45],
            frame_buffer_size=1,
            width=params.width,
            height=params.height,
            warmup=10,
            acceleration=args.acceleration,
            is_drawing=True,
            mode="img2img",
            use_denoising_batch=args.use_denoising_batch,
            cfg_type=args.cfg_type,
        )
        self.last_prompt = default_prompt
        self.stream.prepare(
            prompt=default_prompt,
            negative_prompt=default_negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.2,
        )

    # stream.prepare(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     num_inference_steps=50,
    #     guidance_scale=guidance_scale,
    # )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        generator = torch.manual_seed(params.seed)
        steps = params.steps
        strength = params.strength

        image_tensor = self.stream.preprocess_image(params.image)
        for _ in range(self.stream.batch_size - 1):
            self.stream(image=image_tensor, prompt=params.prompt)

        output_image = self.stream(image=image_tensor, prompt=params.prompt)

        return output_image
