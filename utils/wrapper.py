import gc
import os
import traceback
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from PIL import Image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class StreamDiffusionWrapper:
    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        controlnet_dict: Optional[Dict[str, float]] = None,
        mode: Literal["img2img", "txt2img"] = "img2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        lcm_lora_id: Optional[str] = None,
        HyperSD_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        CM_lora_type: Literal["lcm", "Hyper_SD", "none"] = "Hyper_SD",
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = "engines",
    ):
        """
        Initializes the StreamDiffusionWrapper.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        controlnet_dict : Optional[Dict[str, float]], optional
            The controlnet_dict to load, by default None.
            Keys are the controlnet names and values are the controlnet scales.
            Example: {'controlnet_1' : 0.5 , 'controlnet_2' : 0.7 ,...}
        mode : Literal["img2img", "txt2img"], optional
            txt2img or img2img, by default "img2img".
        output_type : Literal["pil", "pt", "np", "latent"], optional
            The output type of image, by default "pil".
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
            If None, the default LCM-LoRA
            ("latent-consistency/lcm-lora-sdv1-5") will be used.
        HyperSD_lora_id : Optional[str], optional
            The HyperSD_lora_id to load, by default None.
            If None, the default Hyper-SD
            ("ByteDance/Hyper-SD/Hyper-SD15-1step-lora.safetensors") will be used.

            "Hyper_SD_1step": "Hyper-SD15-1step-lora.safetensors"
            "Hyper_SD_2step" : "Hyper-SD15-2steps-lora.safetensors"
            "Hyper_SD_4step" : "Hyper-SD15-4steps-lora.safetensors"
            "Hyper_SD_8step" : "Hyper-SD15-8steps-lora.safetensors"

            Select the Hyper_SD_LoRA_name from the above list
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
            If None, the default TinyVAE
            ("madebyollin/taesd") will be used.
        device : Literal["cpu", "cuda"], optional
            The device to use for inference, by default "cuda".
        dtype : torch.dtype, optional
            The dtype for inference, by default torch.float16.
        frame_buffer_size : int, optional
            The frame buffer size for denoising batch, by default 1.
        width : int, optional
            The width of the image, by default 512.
        height : int, optional
            The height of the image, by default 512.
        acceleration : Literal["none", "xformers", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        device_ids : Optional[List[int]], optional
            The device ids to use for DataParallel, by default None.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        enable_similar_image_filter : bool, optional
            Whether to enable similar image filter or not,
            by default False.
        similar_image_filter_threshold : float, optional
            The threshold for similar image filter, by default 0.98.
        similar_image_filter_max_skip_frame : int, optional
            The max skip frame for similar image filter, by default 10.
        use_denoising_batch : bool, optional
            Whether to use denoising batch or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.
        use_safety_checker : bool, optional
            Whether to use safety checker or not, by default False.
        """
        self.sd_turbo = "turbo" in model_id_or_path

        if mode == "txt2img":
            if cfg_type != "none":
                raise ValueError(f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}")
            if use_denoising_batch and frame_buffer_size > 1:
                if not self.sd_turbo:
                    raise ValueError("txt2img mode cannot use denoising batch with frame_buffer_size > 1.")

        if mode == "img2img":
            if not use_denoising_batch:
                raise NotImplementedError("img2img mode must use denoising batch for now.")

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.mode = mode
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = len(t_index_list) * frame_buffer_size if use_denoising_batch else frame_buffer_size

        self.use_denoising_batch = use_denoising_batch
        self.use_safety_checker = use_safety_checker

        self.is_controlnet_enabled = controlnet_dict is not None

        self.stream: StreamDiffusion = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            controlnet_dict=controlnet_dict,
            lcm_lora_id=lcm_lora_id,
            HyperSD_lora_id=HyperSD_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            do_add_noise=do_add_noise,
            CM_lora_type=CM_lora_type,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
        )

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(self.stream.unet, device_ids=device_ids)

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(
                similar_image_filter_threshold, similar_image_filter_max_skip_frame
            )

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> None:
        """
        Prepares the model for inference.

        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise,
            by default 1.0.
        """
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
        )

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
        controlnet_images: Optional[Union[str, Image.Image, list[str], list[Image.Image], torch.Tensor]] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Performs img2img or txt2img based on the mode.

        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.
        controlnet_images : Optional[Union[str, Image.Image, list[str], list[Image.Image], torch.Tensor]]
            The controlnet image(s) to use for inference if controlnet is enabled.
            by default None.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        assert (self.is_controlnet_enabled and controlnet_images is not None) or (
            not self.is_controlnet_enabled and controlnet_images is None
        ), "If ControlNet is disabled, please do not provide controlnet_images, vice versa."

        if self.mode == "img2img":
            return self.img2img(image, prompt, controlnet_images)
        else:
            return self.txt2img(prompt, controlnet_images)

    def txt2img(
        self,
        prompt: Optional[str] = None,
        controlnet_images: Optional[Union[str, Image.Image, list[str], list[Image.Image], torch.Tensor]] = None,
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs txt2img.

        Parameters
        ----------
        prompt : Optional[str]
            The prompt to generate images from.
        controlnet_images : Optional[Union[str, Image.Image, list[str], list[Image.Image], torch.Tensor]]
            The controlnet image(s) to use for inference if controlnet is enabled.
            by default None.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(controlnet_images, str) or isinstance(controlnet_images, Image.Image):
            controlnet_images = self.preprocess_image(controlnet_images, is_controlnet_image=True)
        elif isinstance(controlnet_images, list):
            controlnet_images = [self.preprocess_image(img, is_controlnet_image=True) for img in controlnet_images]
            controlnet_images = torch.stack(controlnet_images)

        if self.sd_turbo:
            image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
        else:
            image_tensor = self.stream.txt2img(self.frame_buffer_size, controlnet_images)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def img2img(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        prompt: Optional[str] = None,
        controlnet_images: Optional[Union[str, Image.Image, list[str], list[Image.Image], torch.Tensor]] = None,
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs img2img.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.
        controlnet_images : Optional[Union[str, Image.Image, list[str], list[Image.Image], torch.Tensor]]
            The controlnet image(s) to use for inference if controlnet is enabled.

        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        if isinstance(controlnet_images, str) or isinstance(controlnet_images, Image.Image):
            controlnet_images = self.preprocess_image(controlnet_images, is_controlnet_image=True)

        if isinstance(controlnet_images, list):
            controlnet_images = [self.preprocess_image(img, is_controlnet_image=True) for img in controlnet_images]
            controlnet_images = torch.stack(controlnet_images)

        image_tensor = self.stream(image, controlnet_images=controlnet_images)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def preprocess_image(self, image: Union[str, Image.Image], is_controlnet_image: bool = False) -> torch.Tensor:
        """
        Preprocesses the image.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.
        is_controlnet_image : bool, optional
            Whether the image is a control image or not, by default False.

        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return (
            self.stream.image_processor.preprocess(image, self.height, self.width).to(
                device=self.device, dtype=self.dtype
            )
            if not is_controlnet_image
            else self.stream.controlnet_image_processor.preprocess(image, self.height, self.width).to(
                device=self.device, dtype=self.dtype
            )
        )

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
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
            return postprocess_image(image_tensor.cpu(), output_type=output_type)
        else:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)[0]

    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        controlnet_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        HyperSD_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        CM_lora_type: Literal["lcm", "Hyper_SD", "none"] = "lcm",
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = "engines",
    ) -> StreamDiffusion:
        """
        Loads the model.

        This method does the following:

        1. Loads the model from the model_id_or_path.
        2. Loads and fuses the LCM-LoRA model from the lcm_lora_id if needed.
        3. Loads the VAE model from the vae_id if needed.
        4. Enables acceleration if needed.
        5. Prepares the model for inference.
        6. Load the safety checker if needed.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        controlnet_dict : Optional[Dict[str, float]], optional
            The controlnet_dict to load, by default None.
            Keys are the controlnet names and values are the controlnet scales.
            Example: {'controlnet_1' : 0.5 , 'controlnet_2' : 0.7 ,...}
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
        acceleration : Literal["none", "xfomers", "sfast", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.

        Returns
        -------
        StreamDiffusion
            The loaded model.
        """

        try:  # Load from local directory
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
                model_id_or_path,
            ).to(device=self.device, dtype=self.dtype)

        except ValueError:  # Load from huggingface
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(
                model_id_or_path,
            ).to(device=self.device, dtype=self.dtype)
        except Exception:  # No model found
            traceback.print_exc()
            print("Model load has failed. Doesn't exist.")
            exit()

        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
        )
        if not self.sd_turbo:
            if CM_lora_type == "lcm":
                print("-----------------Using lcm-----------------")
                if lcm_lora_id is not None:
                    stream.load_lcm_lora(pretrained_model_name_or_path_or_dict=lcm_lora_id)
                else:
                    stream.load_lcm_lora()
                stream.fuse_lora()

            elif CM_lora_type == "Hyper_SD":
                print(f"-----------------Using Hyper_SD {HyperSD_lora_id}-----------------")
                if HyperSD_lora_id is not None:
                    stream.load_HyperSD_lora(
                        pretrained_model_name_or_path_or_dict="ByteDance/Hyper-SD", model_name=HyperSD_lora_id
                    )
                elif HyperSD_lora_id is None and controlnet_dict is not None:
                    stream.load_HyperSD_lora(
                        pretrained_model_name_or_path_or_dict="ByteDance/Hyper-SD",
                        model_name="Hyper-SD15-4step-lora.safetensors",
                    )
                    print("To generate better results with ControlNet, using 4-steps Hyper-SD instead of 1-step.")
                else:
                    stream.load_HyperSD_lora(
                        pretrained_model_name_or_path_or_dict="ByteDance/Hyper-SD",
                        model_name="Hyper-SD15-1step-lora.safetensors",
                    )
                    print("Using 1-step Hyper-SD.")
                stream.fuse_lora()
            else:  # CM_lora_type == "none"
                pass

            if lora_dict is not None:
                for lora_name, lora_scale in lora_dict.items():
                    stream.load_lora(lora_name)
                    stream.fuse_lora(lora_scale=lora_scale)
                    print(f"Use LoRA: {lora_name} in weights {lora_scale}")

            if controlnet_dict is not None:
                stream.load_controlnet(controlnet_dict)
                print(f"Use controlnet: {controlnet_dict}")

        if use_tiny_vae:
            if vae_id is not None:
                stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(device=pipe.device, dtype=pipe.dtype)
            else:
                stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
                    device=pipe.device, dtype=pipe.dtype
                )

        try:
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
            if acceleration == "tensorrt":
                from polygraphy import cuda

                from streamdiffusion.acceleration.tensorrt import (
                    TorchVAEEncoder,
                    compile_control_unet,
                    compile_unet,
                    compile_vae_decoder,
                    compile_vae_encoder,
                )
                from streamdiffusion.acceleration.tensorrt.engine import (
                    AutoencoderKLEngine,
                    UNet2DConditionControlNetModelEngine,
                    UNet2DConditionModelEngine,
                )
                from streamdiffusion.acceleration.tensorrt.models import (
                    VAE,
                    UNet,
                    UNetWithControlNet,
                    VAEEncoder,
                )

                def create_prefix(
                    model_id_or_path: str,
                    max_batch_size: int,
                    min_batch_size: int,
                ):
                    maybe_path = Path(model_id_or_path)
                    if maybe_path.exists():
                        return f"{maybe_path.stem}--CM_lora_type-{CM_lora_type}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-{self.mode}--controlnet-{'enabled' if self.is_controlnet_enabled else 'disabled'}"
                    else:
                        return f"{model_id_or_path}--CM_lora_type-{CM_lora_type}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-{self.mode}--controlnet-{'enabled' if self.is_controlnet_enabled else 'disabled'}"

                engine_dir = Path(engine_dir)
                unet_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                    ),
                    "unet.engine",
                )
                vae_encoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                        min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    ),
                    "vae_encoder.engine",
                )
                vae_decoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                        min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    ),
                    "vae_decoder.engine",
                )

                if not os.path.exists(unet_path):
                    os.makedirs(os.path.dirname(unet_path), exist_ok=True)
                    if self.is_controlnet_enabled:
                        unet_model = UNetWithControlNet(
                            fp16=True,
                            device=stream.device,
                            max_batch_size=stream.trt_unet_batch_size,
                            min_batch_size=stream.trt_unet_batch_size,
                            num_controlnets=len(controlnet_dict),
                            embedding_dim=stream.text_encoder.config.hidden_size,
                            unet_dim=stream.unet.unet.config.in_channels,
                        )
                        compile_control_unet(
                            stream.unet,
                            unet_model,
                            unet_path + ".onnx",
                            unet_path + ".opt.onnx",
                            unet_path,
                            opt_batch_size=stream.trt_unet_batch_size,
                        )
                    else:
                        unet_model = UNet(
                            fp16=True,
                            device=stream.device,
                            max_batch_size=stream.trt_unet_batch_size,
                            min_batch_size=stream.trt_unet_batch_size,
                            embedding_dim=stream.text_encoder.config.hidden_size,
                            unet_dim=stream.unet.config.in_channels,
                        )
                        compile_unet(
                            stream.unet,
                            unet_model,
                            unet_path + ".onnx",
                            unet_path + ".opt.onnx",
                            unet_path,
                            opt_batch_size=stream.trt_unet_batch_size,
                        )

                if not os.path.exists(vae_decoder_path):
                    os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                    stream.vae.forward = stream.vae.decode
                    vae_decoder_model = VAE(
                        device=stream.device,
                        max_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                        min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    )
                    compile_vae_decoder(
                        stream.vae,
                        vae_decoder_model,
                        vae_decoder_path + ".onnx",
                        vae_decoder_path + ".opt.onnx",
                        vae_decoder_path,
                        opt_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    )
                    delattr(stream.vae, "forward")

                if not os.path.exists(vae_encoder_path):
                    os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
                    vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                    vae_encoder_model = VAEEncoder(
                        device=stream.device,
                        max_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                        min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    )
                    compile_vae_encoder(
                        vae_encoder,
                        vae_encoder_model,
                        vae_encoder_path + ".onnx",
                        vae_encoder_path + ".opt.onnx",
                        vae_encoder_path,
                        opt_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    )

                cuda_steram = cuda.Stream()

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype

                if self.is_controlnet_enabled:
                    stream.unet = UNet2DConditionControlNetModelEngine(unet_path, cuda_steram, use_cuda_graph=False)
                else:
                    stream.unet = UNet2DConditionModelEngine(unet_path, cuda_steram, use_cuda_graph=False)
                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_steram,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                setattr(stream.vae, "config", vae_config)
                setattr(stream.vae, "dtype", vae_dtype)

                gc.collect()
                torch.cuda.empty_cache()

                print("TensorRT acceleration enabled.")
            if acceleration == "sfast":
                from streamdiffusion.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

                stream = accelerate_with_stable_fast(stream)
                print("StableFast acceleration enabled.")
        except Exception as e:
            print(e)
            # traceback.print_exc()
            print("Acceleration has failed. Falling back to normal mode.")

        if seed < 0:  # Random seed
            seed = np.random.randint(0, 1000000)

        stream.prepare(
            "",
            "",
            num_inference_steps=50,
            guidance_scale=1.1 if stream.cfg_type in ["full", "self", "initialize"] else 1.0,
            generator=torch.Generator(),
            seed=seed,
        )

        if self.use_safety_checker:
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )
            from transformers import CLIPFeatureExtractor

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(pipe.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
            self.nsfw_fallback_img = Image.new("RGB", (512, 512), (0, 0, 0))

        return stream
