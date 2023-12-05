from typing import *

import numpy as np
import PIL.Image
import torch
from diffusers import LCMScheduler, StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

from streamdiffusion.image_filter import SimilarImageFilter


class StreamDiffusion:
    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        t_index_list: List[int],
        torch_dtype=torch.float16,
        width: int = 512,
        height: int = 512,
        is_drawing: bool = False,
        frame_buffer_size: int = 1,
    ):
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size

        self.batch_size = len(t_index_list) * frame_buffer_size
        self.t_list = t_index_list

        self.is_drawing = is_drawing

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[
            str, Dict[str, torch.Tensor]
        ] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name=None,
        **kwargs,
    ):
        self.pipe.load_lora_weights(pretrained_model_name_or_path_or_dict, adapter_name, **kwargs)

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ):
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.95):
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)

    def disable_similar_image_filter(self):
        self.similar_image_filter = False

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = torch.Generator(),
    ):
        self.generator = generator
        # initialize x_t_latent (it can be any random tensor)
        if self.batch_size > 1:
            self.x_t_latent_buffer = torch.zeros(
                (self.batch_size - self.frame_bff_size, 4, self.latent_height, self.latent_width),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None

        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(self.sub_timesteps, dtype=torch.long, device=self.device)
        self.sub_timesteps_tensor = torch.repeat_interleave(sub_timesteps_tensor, repeats=self.frame_bff_size, dim=0)

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)
        c_skip = torch.stack(c_skip_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)
        c_out = torch.stack(c_out_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)
        self.c_skip = torch.repeat_interleave(c_skip, repeats=self.frame_bff_size, dim=0)
        self.c_out = torch.repeat_interleave(c_out, repeats=self.frame_bff_size, dim=0)

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(alpha_prod_t_sqrt, repeats=self.frame_bff_size, dim=0)
        self.beta_prod_t_sqrt = torch.repeat_interleave(beta_prod_t_sqrt, repeats=self.frame_bff_size, dim=0)

    @torch.no_grad()
    def update_prompt(self, prompt: str):
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

    def add_noise(self, original_samples, noise, t_index):
        noisy_samples = self.alpha_prod_t_sqrt[t_index] * original_samples + self.beta_prod_t_sqrt[t_index] * noise
        return noisy_samples

    def scheduler_step_batch(self, model_pred_batch, x_t_latent_batch):
        F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch) / self.alpha_prod_t_sqrt
        denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        return denoised_batch

    def lcm_step(
        self,
        x_t_latent: torch.FloatTensor,
    ):
        model_pred = self.unet(
            x_t_latent,
            self.sub_timesteps_tensor,
            encoder_hidden_states=self.prompt_embeds,
        )[0]

        # compute the previous noisy sample x_t -> x_t-1
        denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent)

        return denoised_batch, model_pred

    def encode_image(self, image_tensors: torch.Tensor):
        image_tensors = image_tensors.to(
            device=self.device,
            dtype=self.vae.dtype,
        )
        img_latent = retrieve_latents(self.vae.encode(image_tensors), self.generator)
        img_latent = img_latent * self.vae.config.scaling_factor
        x_t_latent = self.add_noise(img_latent, self.init_noise[0], 0)
        return x_t_latent

    def decode_image(self, x_0_pred_out):
        output_latent = self.vae.decode(x_0_pred_out / self.vae.config.scaling_factor, return_dict=False)[0]
        return output_latent

    def predict_x0_batch(self, x_t_latent):
        prev_latent_batch = self.x_t_latent_buffer
        if self.batch_size > 1:
            x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)

        x_0_pred_batch, model_pred = self.lcm_step(x_t_latent)
        if prev_latent_batch is not None:
            x_0_pred_out = x_0_pred_batch[-self.frame_bff_size :]
            if self.is_drawing:
                self.x_t_latent_buffer = (
                    self.alpha_prod_t_sqrt[self.frame_bff_size :] * x_0_pred_batch[: -self.frame_bff_size]
                    + self.beta_prod_t_sqrt[self.frame_bff_size :] * self.init_noise[self.frame_bff_size :]
                )
            else:
                self.x_t_latent_buffer = (
                    self.alpha_prod_t_sqrt[self.frame_bff_size :] * x_0_pred_batch[: -self.frame_bff_size]
                )
        else:
            x_0_pred_out = x_0_pred_batch
            self.x_t_latent_buffer = None
        return x_0_pred_out

    @torch.no_grad()
    def __call__(self, x: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray]):
        assert x.shape[0] == self.frame_bff_size, "Input batch size must be equal to frame buffer size."
        x = self.image_processor.preprocess(x, self.height, self.width).to(device=self.device, dtype=self.dtype)
        if self.similar_image_filter:
            x = self.similar_filter(x)
            if x is None:
                return self.prev_image_result

        x_t_latent = self.encode_image(x)
        x_0_pred_out = self.predict_x0_batch(x_t_latent)
        x_output = self.decode_image(x_0_pred_out).detach().clone()

        self.prev_image_result = x_output

        return x_output

    @torch.no_grad()
    def txt2img(self):
        x_0_pred_out = self.predict_x0_batch(
            torch.randn((1, 4, self.latent_height, self.latent_width)).to(device=self.device, dtype=self.dtype)
        )
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        return x_output
