import os
from typing import List, Literal, Optional, Union

import torch
from diffusers import (
    AutoencoderTiny,
    StableDiffusionPipeline,
)
from PIL import Image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


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
        accerelation: Literal["none", "xformers", "sfast", "tensorrt"] = "tensorrt",
        is_drawing: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.95,
    ):
        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = len(t_index_list) * frame_buffer_size

        self.stream = self._load_model(
            model_id=model_id,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            accerelation=accerelation,
            warmup=warmup,
            is_drawing=is_drawing,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
        )

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(
                self.stream.unet, device_ids=device_ids
            )

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(similar_image_filter_threshold)

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
        self.stream.prepare(
            prompt,
            num_inference_steps=num_inference_steps,
        )

    def txt2img(self) -> Union[Image.Image, List[Image.Image]]:
        """
        Performs txt2img.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if self.frame_buffer_size > 1:
            image_tensor = self.stream.txt2img_batch(self.batch_size)
        else:
            image_tensor = self.stream.txt2img()
        return self._postprocess_image(image_tensor)

    def img2img(self, image: Union[str, Image.Image, torch.Tensor]) -> Image.Image:
        """
        Performs img2img.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.

        Returns
        -------
        Image.Image
            The generated image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        image_tensor = self.stream(image)
        return self._postprocess_image(image_tensor)

    def _postprocess_image(
        self, image_tensor: torch.Tensor
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Postprocesses the image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The postprocessed image.
        """
        if self.frame_buffer_size > 1:
            return postprocess_image(image_tensor.cpu(), output_type="pil")
        else:
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
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
    ):
        if model_id.endswith(".safetensors"):
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
            width=self.width,
            height=self.height,
            is_drawing=is_drawing,
            frame_buffer_size=self.frame_buffer_size,
        )
        if "turbo" not in model_id:
            if use_lcm_lora:
                if lcm_lora_id is not None:
                    stream.load_lcm_lora(
                        pretrained_model_name_or_path_or_dict=lcm_lora_id
                    )
                else:
                    stream.load_lcm_lora()
                stream.fuse_lora()

        if use_tiny_vae:
            if vae_id is not None:
                stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(
                    device=pipe.device, dtype=pipe.dtype
                )
            else:
                stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
                    device=pipe.device, dtype=pipe.dtype
                )

        try:
            if accerelation == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
            if accerelation == "tensorrt":
                from streamdiffusion.acceleration.tensorrt import (
                    accelerate_with_tensorrt,
                )

                stream = accelerate_with_tensorrt(
                    stream,
                    os.path.join(
                        CURRENT_DIR,
                        "..",
                        "engines",
                        f"{model_id.replace('/', '_')}_max_batch_{self.batch_size}_min_batch_{self.batch_size}",
                    ),
                    min_batch_size=self.batch_size,
                    max_batch_size=self.batch_size,
                )
                print("TensorRT acceleration enabled.")
            if accerelation == "sfast":
                from streamdiffusion.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

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
            if self.frame_buffer_size > 1:
                self.stream.txt2img_batch(self.batch_size)
            else:
                self.stream.txt2img()
            torch.cuda.synchronize()

        return stream
