from typing import *

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)


class StreamDiffusion:
    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        t_index_list: List[int],
        torch_dtype=torch.float16,
        width: int = 512,
        height: int = 512,
        is_drawing: bool = False,
    ):
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.batch_size = len(t_index_list)
        self.t_list = t_index_list

        self.is_drawing = is_drawing

        self.pipe = pipe

        self.scheduler = pipe.scheduler
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
    ):
        self.generator = generator
        # initialize x_t_latent (it can be any random tensor)
        if self.batch_size > 1:
            self.x_t_latent_buffer = torch.zeros(
                (self.batch_size - 1, 4, self.latent_height, self.latent_width),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None

        self.additional_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.prompt_embeds = self.additional_embeds[0].repeat(self.batch_size, 1, 1)

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        self.sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)
        self.c_skip = (
            torch.stack(c_skip_list)
            .view(self.batch_size, 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(self.batch_size, 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        self.alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(self.batch_size, 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(self.batch_size, 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

    def add_noise(self, original_samples, noise, t_index):
        noisy_samples = (
            self.alpha_prod_t_sqrt[t_index] * original_samples
            + self.beta_prod_t_sqrt[t_index] * noise
        )
        return noisy_samples

    def scheduler_step_batch(self, model_pred_batch, x_t_latent_batch):
        F_theta = (
            x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch
        ) / self.alpha_prod_t_sqrt
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
        output_latent = self.vae.decode(
            x_0_pred_out / self.vae.config.scaling_factor, return_dict=False
        )[0]
        return output_latent

    def predict_x0_batch(self, x_t_latent):
        prev_latent_batch = self.x_t_latent_buffer
        if self.batch_size > 1:
            x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)

        x_0_pred_batch, model_pred = self.lcm_step(x_t_latent)
        if prev_latent_batch is not None:
            x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
            if self.is_drawing:
                self.x_t_latent_buffer = (
                    self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                    + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                )
            else:
                self.x_t_latent_buffer = (
                    self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                )
        else:
            x_0_pred_out = x_0_pred_batch
            self.x_t_latent_buffer = None
        return x_0_pred_out

    @torch.no_grad()
    def __call__(self, x: torch.Tensor):
        x_t_latent = self.encode_image(x)
        x_0_pred_out = self.predict_x0_batch(x_t_latent)
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        return x_output

    @torch.no_grad()
    def txt2img(self, prompt: str):
        self.additional_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.prompt_embeds = self.additional_embeds[0].repeat(self.batch_size, 1, 1)
        x_0_pred_out = self.predict_x0_batch(
            torch.randn((1, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
        )
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        return x_output
