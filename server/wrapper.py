import io
from typing import List

import PIL.Image
import requests
import torch
from diffusers import AutoencoderTiny, LCMScheduler, StableDiffusionPipeline
from streamdiffusion import StreamDiffusion
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
        pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            model_id
        ).to(device=self.device, dtype=self.dtype)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.vae = AutoencoderTiny.from_pretrained(vae_id).to(
            device=pipe.device, dtype=pipe.dtype
        )
        pipe.load_lora_weights(lcm_lora_id)
        pipe.fuse_lora()
        pipe.enable_xformers_memory_efficient_attention()

        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            is_drawing=True,
        )

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
            stream.txt2img("")
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
            x_output = self.stream.txt2img(prompt)
            if i >= 3:
                images.append(postprocess_image(x_output, output_type="pil")[0])
            end.record()

            torch.cuda.synchronize()

        return images


if __name__ == "__main__":
    wrapper = StreamDiffusionWrapper(10, 10)
    wrapper()
