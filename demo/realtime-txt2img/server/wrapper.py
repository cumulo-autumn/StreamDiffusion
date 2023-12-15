import io
import os
from typing import List

import PIL.Image
import requests
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

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
        safety_checker: bool,
    ):
        self.device = device
        self.dtype = dtype
        self.prompt = ""
        self.batch_size = len(t_index_list)

        self.stream = self._load_model(
            model_id=model_id,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            warmup=warmup,
        )
        self.safety_checker = None
        if safety_checker:
            from transformers import CLIPFeatureExtractor
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(self.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.nsfw_fallback_img = PIL.Image.new("RGB", (512, 512), (0, 0, 0))
        self.stream.prepare("")

    def _load_model(
        self,
        model_id: str,
        lcm_lora_id: str,
        vae_id: str,
        t_index_list: List[int],
        warmup: int,
    ):
        if os.path.exists(model_id):
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(
                model_id
            ).to(device=self.device, dtype=self.dtype)
        else:
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
                model_id
            ).to(device=self.device, dtype=self.dtype)

        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            is_drawing=True,
        )
        stream.load_lcm_lora(lcm_lora_id)
        stream.fuse_lora()
        stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(
            device=pipe.device, dtype=pipe.dtype
        )

        try:
            from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

            stream = accelerate_with_tensorrt(
                stream,
                "engines",
                max_batch_size=self.batch_size,
                engine_build_options={"build_static_batch": False},
            )
            print("TensorRT acceleration enabled.")
        except Exception:
            print("TensorRT acceleration has failed. Trying to use Stable Fast.")
            try:
                from streamdiffusion.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

                stream = accelerate_with_stable_fast(stream)
                print("StableFast acceleration enabled.")
            except Exception:
                print("StableFast acceleration has failed. Using normal mode.")
                pass

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

    def __call__(self, prompt: str) -> PIL.Image.Image:
        if self.prompt != prompt:
            self.stream.update_prompt(prompt)
            self.prompt = prompt
            for i in range(self.batch_size):
                x_output = self.stream.txt2img()

        x_output = self.stream.txt2img()
        image = postprocess_image(x_output, output_type="pil")[0]

        if self.safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=x_output,
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image


if __name__ == "__main__":
    wrapper = StreamDiffusionWrapper(10, 10)
    wrapper()
