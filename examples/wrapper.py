import os
from typing import List, Literal, Optional

import torch
from diffusers import (
    AutoencoderTiny,
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
)
from PIL import Image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import pil2tensor, postprocess_image

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class StreamDiffusionWrapper:
    def __init__(
        self,
        model_id: str,
        t_index_list: List[int],
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        accerelation: Literal["none", "sfast", "tensorrt"] = "tensorrt",
        is_drawing: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.prompt = ""
        self.width = width
        self.height = height
        self.batch_size = len(t_index_list) * frame_buffer_size

        self.stream = self._load_model(
            model_id=model_id,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            accerelation=accerelation,
            warmup=warmup,
            is_drawing=is_drawing,
        )

    def prepare(
        self,
        prompt: str,
        num_inference_steps: int = 50,
    ) -> None:
        """
        Prepares the model for inference.

        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        """
        self.prompt = prompt
        self.stream.prepare(
            prompt,
            num_inference_steps=num_inference_steps,
        )

    def img2img(self, image_path: str) -> Image.Image:
        """
        Performs img2img.

        Parameters
        ----------
        image_path : str
            The image to generate from.

        Returns
        -------
        Image.Image
            The generated image.
        """
        image_tensor = self.stream(Image.open(image_path).convert("RGB").resize((self.width, self.height)))
        return self._postprocess_image(image_tensor)

    def _postprocess_image(self, image_tensor: torch.Tensor) -> Image.Image:
        """
        Postprocesses the image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.

        Returns
        -------
        Image.Image
            The postprocessed image.
        """
        return postprocess_image(image_tensor.cpu(), output_type="pil")[0]

    def _load_model(
        self,
        model_id: str,
        t_index_list: List[int],
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        accerelation: Literal["none", "sfast", "tensorrt"] = "tensorrt",
        warmup: int = 10,
        is_drawing: bool = True,
    ):
        if model_id.endswith(".safetensors"):
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(model_id).to(
                device=self.device, dtype=self.dtype
            )
        else:
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(model_id).to(
                device=self.device, dtype=self.dtype
            )
        pipe.enable_xformers_memory_efficient_attention()
        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            is_drawing=is_drawing,
        )
        if lcm_lora_id is not None:
            stream.load_lcm_lora(pretrained_model_name_or_path_or_dict=lcm_lora_id)
        else:
            stream.load_lcm_lora()

        stream.fuse_lora()
        if vae_id is not None:
            stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(device=pipe.device, dtype=pipe.dtype)
        else:
            stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

        try:
            if accerelation == "tensorrt":
                from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
                stream = accelerate_with_tensorrt(
                    stream, "engines",
                    min_batch_size=self.batch_size,
                    max_batch_size=self.batch_size,
                    engine_build_options={"build_static_batch": False}
                )
                print("TensorRT acceleration enabled.")
            elif accerelation == "sfast":
                from streamdiffusion.acceleration.sfast import accelerate_with_stable_fast
                stream = accelerate_with_stable_fast(stream)
                print("StableFast acceleration enabled.")
        except Exception:
            print("Acceleration has failed. Falling back to normal mode.")

        stream.prepare(
            "",
            num_inference_steps=50,
            generator=torch.manual_seed(2),
        )

        # warmup
        for _ in range(warmup):
            stream.txt2img()
            torch.cuda.synchronize()

        return stream
