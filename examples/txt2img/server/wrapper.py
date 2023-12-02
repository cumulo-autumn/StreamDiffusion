import io
import os
from typing import List

import PIL.Image
import requests
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.sfast import accelerate_with_stable_fast
from streamdiffusion.image_utils import postprocess_image


def download_image(url: str):
    response = requests.get(url)
    image = PIL.Image.open(io.BytesIO(response.content))
    return image


class StreamDiffusionWrapper:
    def __init__(
        self,
        model_id: str,
        lcm_lora_id: str,
        vae_id: str,
        device: str,
        dtype: str,
        t_index_list: List[int],
        warmup: int,
    ):
        self.device = device
        self.dtype = dtype
        self.prompt = ""

        self.stream = self._load_model(
            model_id=model_id,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            warmup=warmup,
        )

    def _load_model(
        self,
        model_id: str,
        lcm_lora_id: str,
        vae_id: str,
        t_index_list: List[int],
        warmup: int,
    ):
        if os.path.exists(model_id):
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(model_id).to(
                device=self.device, dtype=self.dtype
            )
        else:
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(model_id).to(
                device=self.device, dtype=self.dtype
            )

        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            is_drawing=True,
        )
        stream.load_lcm_lora(lcm_lora_id)
        stream.fuse_lora()
        stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(device=pipe.device, dtype=pipe.dtype)
        stream = accelerate_with_stable_fast(stream)

        stream.prepare(
            "",
            num_inference_steps=50,
            generator=torch.manual_seed(2),
        )

        # warmup
        for _ in range(warmup):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            stream.txt2img()
            end.record()

            torch.cuda.synchronize()

        return stream

    def __call__(self, prompt: str) -> List[PIL.Image.Image]:
        self.stream.prepare("")

        images = []
        for i in range(9 + 3):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            if self.prompt != prompt:
                self.stream.update_prompt(prompt)
                self.prompt = prompt

            x_output = self.stream.txt2img()
            if i >= 3:
                images.append(postprocess_image(x_output, output_type="pil")[0])
            end.record()

            torch.cuda.synchronize()

        return images


if __name__ == "__main__":
    wrapper = StreamDiffusionWrapper(10, 10)
    wrapper()
