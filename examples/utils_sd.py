import imp
import numpy as np
import cv2
import torch
import random
from PIL import Image, ImageDraw, ImageFont
import copy
from typing import Optional, Union, Tuple, List, Callable, Dict, Any
from tqdm.notebook import tqdm
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
    get_down_block,
    get_up_block,
)
from diffusers.models.unet_2d_condition import UNet2DConditionOutput, logger
from copy import deepcopy
import json

import inspect
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import is_compiled_module

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from tqdm import tqdm
# from controlnet_aux import HEDdetector, OpenposeDetector
import time

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_promptls(prompt_path):
    with open(prompt_path) as f:
        prompt_ls = json.load(f)
    prompt_ls = [prompt['caption'].replace('/','_') for prompt in prompt_ls]
    return prompt_ls

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    # print(image_path)
    if type(image_path) is str:
        image = np.array(Image.open(image_path))
        if image.ndim>3:
            image = image[:,:,:3]
        elif image.ndim == 2:
            image = image.reshape(image.shape[0], image.shape[1],1).astype('uint8')
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def get_canny(image_path):
    image = load_512(
        image_path
    )
    image = np.array(image)

    # get canny image
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def get_scribble(image_path, hed):
    image = load_512(
        image_path
    )
    image = hed(image, scribble=True)

    return image

def get_cocoimages(prompt_path):
    data_ls = []
    with open(prompt_path) as f:
        prompt_ls = json.load(f)
    img_path = 'COCO2017-val/val2017'
    for prompt in tqdm(prompt_ls):
        caption = prompt['caption'].replace('/','_')
        image_id = str(prompt['image_id'])
        image_id = (12-len(image_id))*'0' + image_id+'.jpg'
        image_path = os.path.join(img_path, image_id)
        try:
            image = get_canny(image_path)
        except:
            continue
        curr_data = {'image':image, 'prompt':caption}
        data_ls.append(curr_data)
    return data_ls

def get_cocoimages2(prompt_path):
    """scribble condition
    """
    data_ls = []
    with open(prompt_path) as f:
        prompt_ls = json.load(f)
    img_path = 'COCO2017-val/val2017'
    hed = HEDdetector.from_pretrained('ControlNet/detector_weights/annotator', filename='network-bsds500.pth')
    for prompt in tqdm(prompt_ls):
        caption = prompt['caption'].replace('/','_')
        image_id = str(prompt['image_id'])
        image_id = (12-len(image_id))*'0' + image_id+'.jpg'
        image_path = os.path.join(img_path, image_id)
        try:
            image = get_scribble(image_path,hed)
        except:
            continue
        curr_data = {'image':image, 'prompt':caption}
        data_ls.append(curr_data)
    return data_ls

def warpped_feature(sample, step):
    """
    sample: batch_size*dim*h*w, uncond: 0 - batch_size//2, cond: batch_size//2 - batch_size
    step: timestep span
    """
    bs, dim, h, w = sample.shape
    uncond_fea, cond_fea = sample.chunk(2)
    uncond_fea = uncond_fea.repeat(step,1,1,1) # (step * bs//2) * dim * h *w
    cond_fea = cond_fea.repeat(step,1,1,1) # (step * bs//2) * dim * h *w
    return torch.cat([uncond_fea, cond_fea])

def warpped_skip_feature(block_samples, step):
    down_block_res_samples = []
    for sample in block_samples:
        sample_expand = warpped_feature(sample, step)
        down_block_res_samples.append(sample_expand)
    return tuple(down_block_res_samples)

def warpped_text_emb(text_emb, step):
    """
    text_emb: batch_size*77*768, uncond: 0 - batch_size//2, cond: batch_size//2 - batch_size
    step: timestep span
    """
    bs, token_len, dim = text_emb.shape
    uncond_fea, cond_fea = text_emb.chunk(2)
    uncond_fea = uncond_fea.repeat(step,1,1) # (step * bs//2) * 77 *768
    cond_fea = cond_fea.repeat(step,1,1) # (step * bs//2) * 77 * 768
    return torch.cat([uncond_fea, cond_fea]) # (step*bs) * 77 *768

def warpped_timestep(timesteps, bs):
    """
    timestpes: list, such as [981, 961, 941]
    """
    semi_bs = bs//2
    ts = []
    for timestep in timesteps:
        timestep = timestep[None]
        texp = timestep.expand(semi_bs)
        ts.append(texp)
    timesteps = torch.cat(ts)
    return timesteps.repeat(2,1).reshape(-1)

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def register_normal_pipeline(pipe):
    def new_call(self):
        @torch.no_grad()
        def call(
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            **kwargs,
        ):

            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)


            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor
            # to deal with lora scaling and other possible forward hooks

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._guidance_rescale = guidance_rescale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            # 3. Encode input prompt
            lora_scale = (
                self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
            )

            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 6.5 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)
            init_latents = latents.detach().clone()
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if t/1000 < 0.5:
                        latents = latents + 0.003*init_latents
                    setattr(self.unet, 'order', i)
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

            if not output_type == "latent":
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                    0
                ]
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        return call
    pipe.call = new_call(pipe)


def register_parallel_pipeline(pipe):
    def new_call(self):
        @torch.no_grad()
        def call(
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            **kwargs,
        ):

            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)


            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor
            # to deal with lora scaling and other possible forward hooks

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._guidance_rescale = guidance_rescale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            # 3. Encode input prompt
            lora_scale = (
                self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
            )

            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 6.5 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)
            init_latents = latents.detach().clone()
            #-------------------------------------------------------
            all_steps = len(self.scheduler.timesteps)
            curr_span = 1
            curr_step = 0

            # st = time.time()
            idx = 1
            keytime = [0,1,2,3,5,10,15,25,35]
            keytime.append(all_steps)
            while curr_step<all_steps:
                refister_time(self.unet, curr_step)

                merge_span = curr_span
                if merge_span>0:
                    time_ls = []
                    for i in range(curr_step, curr_step+merge_span):
                        if i<all_steps:
                            time_ls.append(self.scheduler.timesteps[i])
                        else:
                            break

                    ##--------------------------------
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        time_ls,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1

                    step_span = len(time_ls)
                    bs = noise_pred.shape[0]
                    bs_perstep = bs//step_span

                    denoised_latent = latents
                    for i, timestep in enumerate(time_ls):
                        if timestep/1000 < 0.5:
                            denoised_latent = denoised_latent + 0.003*init_latents
                        curr_noise = noise_pred[i*bs_perstep:(i+1)*bs_perstep]
                        denoised_latent = self.scheduler.step(curr_noise, timestep, denoised_latent, **extra_step_kwargs, return_dict=False)[0]
                    
                    latents = denoised_latent
                    ##----------------------------------------
                curr_step += curr_span
                idx += 1

                if curr_step<all_steps:
                    curr_span = keytime[idx] - keytime[idx-1] 

           
            if not output_type == "latent":
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                    0
                ]
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        return call
    pipe.call = new_call(pipe)

def register_faster_forward(model, mod = '50ls'):
    def faster_forward(self):
        def forward(
                sample: torch.FloatTensor,
                timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states: torch.Tensor,
                class_labels: Optional[torch.Tensor] = None,
                timestep_cond: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
                mid_block_additional_residual: Optional[torch.Tensor] = None,
                return_dict: bool = True,
            ) -> Union[UNet2DConditionOutput, Tuple]:
                r"""
                Args:
                    sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
                    timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
                    encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
                    return_dict (`bool`, *optional*, defaults to `True`):
                        Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
                    cross_attention_kwargs (`dict`, *optional*):
                        A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                        `self.processor` in
                        [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

                Returns:
                    [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                    [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
                    returning a tuple, the first element is the sample tensor.
                """
                # By default samples have to be AT least a multiple of the overall upsampling factor.
                # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
                # However, the upsampling interpolation output size can be forced to fit any upsampling size
                # on the fly if necessary.
                default_overall_up_factor = 2**self.num_upsamplers

                # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
                forward_upsample_size = False
                upsample_size = None

                if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
                    logger.info("Forward upsample size to force interpolation output size.")
                    forward_upsample_size = True

                # prepare attention_mask
                if attention_mask is not None:
                    attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                    attention_mask = attention_mask.unsqueeze(1)

                # 0. center input if necessary
                if self.config.center_input_sample:
                    sample = 2 * sample - 1.0

                # 1. time
                if isinstance(timestep, list):
                    timesteps = timestep[0]
                    step = len(timestep)
                else:
                    timesteps = timestep
                    step = 1
                if not torch.is_tensor(timesteps) and (not isinstance(timesteps,list)):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = sample.device.type == "mps"
                    if isinstance(timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
                elif (not isinstance(timesteps,list)) and len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(sample.device)
                
                if (not isinstance(timesteps,list)) and len(timesteps.shape) == 1:
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timesteps = timesteps.expand(sample.shape[0])
                elif isinstance(timesteps, list):
                    #timesteps list, such as [981,961,941]
                    timesteps = warpped_timestep(timesteps, sample.shape[0]).to(sample.device)
                t_emb = self.time_proj(timesteps)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # but time_embedding might actually be running in fp16. so we need to cast here.
                # there might be better ways to encapsulate this.
                t_emb = t_emb.to(dtype=self.dtype)

                emb = self.time_embedding(t_emb, timestep_cond)

                if self.class_embedding is not None:
                    if class_labels is None:
                        raise ValueError("class_labels should be provided when num_class_embeds > 0")

                    if self.config.class_embed_type == "timestep":
                        class_labels = self.time_proj(class_labels)

                        # `Timesteps` does not contain any weights and will always return f32 tensors
                        # there might be better ways to encapsulate this.
                        class_labels = class_labels.to(dtype=sample.dtype)

                    class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

                    if self.config.class_embeddings_concat:
                        emb = torch.cat([emb, class_emb], dim=-1)
                    else:
                        emb = emb + class_emb

                if self.config.addition_embed_type == "text":
                    aug_emb = self.add_embedding(encoder_hidden_states)
                    emb = emb + aug_emb

                if self.time_embed_act is not None:
                    emb = self.time_embed_act(emb)

                if self.encoder_hid_proj is not None:
                    encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

                #===============
                # order = self.order #timestep, start by 0
                order = 5
                #===============
                ipow = int(np.sqrt(9 + 8*order))
                cond = order in [0, 1, 2, 3, 5, 10, 15, 25, 35]
                if isinstance(mod, int):
                    cond = order % mod == 0
                elif mod == "pro":
                    cond = ipow * ipow == (9 + 8 * order)
                elif mod == "50ls":
                    cond = order in [0, 1, 2, 3, 5, 10, 15, 25, 35] #40 #[0,1,2,3, 5, 10, 15] #[0, 1, 2, 3, 5, 10, 15, 25, 35, 40]
                elif mod == "50ls2":
                    cond = order in [0, 10, 11, 12, 15, 20, 25, 30,35,45] #40 #[0,1,2,3, 5, 10, 15] #[0, 1, 2, 3, 5, 10, 15, 25, 35, 40]
                elif mod == "50ls3":
                    cond = order in [0, 20, 25, 30,35,45,46,47,48,49] #40 #[0,1,2,3, 5, 10, 15] #[0, 1, 2, 3, 5, 10, 15, 25, 35, 40]
                elif mod == "50ls4":
                    cond = order in [0, 9, 13, 14, 15, 28, 29, 32, 36,45] #40 #[0,1,2,3, 5, 10, 15] #[0, 1, 2, 3, 5, 10, 15, 25, 35, 40]
                elif mod == "100ls":
                    cond = order > 85 or order < 10 or order % 5 == 0
                elif mod == "75ls":
                    cond = order > 65 or order < 10 or order % 5 == 0
                elif mod == "s2":
                    cond = order < 20 or order > 40 or order % 2 == 0

                if cond:
                    print(order)
                    # 2. pre-process
                    sample = self.conv_in(sample)

                    # 3. down
                    down_block_res_samples = (sample,)
                    for downsample_block in self.down_blocks:
                        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                            sample, res_samples = downsample_block(
                                hidden_states=sample,
                                temb=emb,
                                encoder_hidden_states=encoder_hidden_states,
                                attention_mask=attention_mask,
                                cross_attention_kwargs=cross_attention_kwargs,
                            )
                        else:
                            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                        down_block_res_samples += res_samples

                    if down_block_additional_residuals is not None:
                        new_down_block_res_samples = ()

                        for down_block_res_sample, down_block_additional_residual in zip(
                            down_block_res_samples, down_block_additional_residuals
                        ):
                            down_block_res_sample = down_block_res_sample + down_block_additional_residual
                            new_down_block_res_samples += (down_block_res_sample,)

                        down_block_res_samples = new_down_block_res_samples

                    # 4. mid
                    if self.mid_block is not None:
                        sample = self.mid_block(
                            sample,
                            emb,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                            cross_attention_kwargs=cross_attention_kwargs,
                        )

                    if mid_block_additional_residual is not None:
                        sample = sample + mid_block_additional_residual

                    #----------------------save feature-------------------------
                    # setattr(self, 'skip_feature', (tmp_sample.clone() for tmp_sample in down_block_res_samples))
                    setattr(self, 'skip_feature', deepcopy(down_block_res_samples))
                    setattr(self, 'toup_feature', sample.detach().clone())
                    #-----------------------save feature------------------------



                    #-------------------expand feature for parallel---------------
                    if isinstance(timestep, list):
                        #timesteps list, such as [981,961,941]
                        timesteps = warpped_timestep(timestep, sample.shape[0]).to(sample.device)
                        t_emb = self.time_proj(timesteps)

                        # `Timesteps` does not contain any weights and will always return f32 tensors
                        # but time_embedding might actually be running in fp16. so we need to cast here.
                        # there might be better ways to encapsulate this.
                        t_emb = t_emb.to(dtype=self.dtype)

                        emb = self.time_embedding(t_emb, timestep_cond)
                        # print(emb.shape)

                    # print(step, sample.shape)
                    down_block_res_samples = warpped_skip_feature(down_block_res_samples, step)
                    sample = warpped_feature(sample, step)
                    # print(step, sample.shape)

                    encoder_hidden_states = warpped_text_emb(encoder_hidden_states, step)

                    # print(emb.shape)

                    #-------------------expand feature for parallel---------------
                    
                else:
                    down_block_res_samples = self.skip_feature
                    sample = self.toup_feature

                    #-------------------expand feature for parallel---------------
                    down_block_res_samples = warpped_skip_feature(down_block_res_samples, step)
                    sample = warpped_feature(sample, step)
                    encoder_hidden_states = warpped_text_emb(encoder_hidden_states, step)
                    #-------------------expand feature for parallel---------------

                # 5. up
                for i, upsample_block in enumerate(self.up_blocks):
                    is_final_block = i == len(self.up_blocks) - 1

                    res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                    down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                    # if we have not reached the final block and need to forward the
                    # upsample size, we do it here
                    if not is_final_block and forward_upsample_size:
                        upsample_size = down_block_res_samples[-1].shape[2:]

                    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                        )

                # 6. post-process
                if self.conv_norm_out:
                    sample = self.conv_norm_out(sample)
                    sample = self.conv_act(sample)
                sample = self.conv_out(sample)

                if not return_dict:
                    return (sample,)

                return UNet2DConditionOutput(sample=sample)
        return forward
    if model.__class__.__name__ == 'UNet2DConditionModel':
        model.forward = faster_forward(model)

def register_normal_forward(model):
    def normal_forward(self):
        def forward(
                sample: torch.FloatTensor,
                timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states: torch.Tensor,
                class_labels: Optional[torch.Tensor] = None,
                timestep_cond: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
                mid_block_additional_residual: Optional[torch.Tensor] = None,
                return_dict: bool = True,
            ) -> Union[UNet2DConditionOutput, Tuple]:
                # By default samples have to be AT least a multiple of the overall upsampling factor.
                # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
                # However, the upsampling interpolation output size can be forced to fit any upsampling size
                # on the fly if necessary.
                default_overall_up_factor = 2**self.num_upsamplers

                # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
                forward_upsample_size = False
                upsample_size = None
                #---------------------
                # import os
                # os.makedirs(f'{timestep.item()}_step', exist_ok=True)
                #---------------------
                if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
                    logger.info("Forward upsample size to force interpolation output size.")
                    forward_upsample_size = True

                # prepare attention_mask
                if attention_mask is not None:
                    attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                    attention_mask = attention_mask.unsqueeze(1)

                # 0. center input if necessary
                if self.config.center_input_sample:
                    sample = 2 * sample - 1.0

                # 1. time
                timesteps = timestep
                if not torch.is_tensor(timesteps):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = sample.device.type == "mps"
                    if isinstance(timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
                elif len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(sample.device)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(sample.shape[0])

                t_emb = self.time_proj(timesteps)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # but time_embedding might actually be running in fp16. so we need to cast here.
                # there might be better ways to encapsulate this.
                t_emb = t_emb.to(dtype=self.dtype)

                emb = self.time_embedding(t_emb, timestep_cond)

                if self.class_embedding is not None:
                    if class_labels is None:
                        raise ValueError("class_labels should be provided when num_class_embeds > 0")

                    if self.config.class_embed_type == "timestep":
                        class_labels = self.time_proj(class_labels)

                        # `Timesteps` does not contain any weights and will always return f32 tensors
                        # there might be better ways to encapsulate this.
                        class_labels = class_labels.to(dtype=sample.dtype)

                    class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

                    if self.config.class_embeddings_concat:
                        emb = torch.cat([emb, class_emb], dim=-1)
                    else:
                        emb = emb + class_emb

                if self.config.addition_embed_type == "text":
                    aug_emb = self.add_embedding(encoder_hidden_states)
                    emb = emb + aug_emb

                if self.time_embed_act is not None:
                    emb = self.time_embed_act(emb)

                if self.encoder_hid_proj is not None:
                    encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

                # 2. pre-process
                sample = self.conv_in(sample)

                # 3. down
                down_block_res_samples = (sample,)
                for i, downsample_block in enumerate(self.down_blocks):
                    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                        sample, res_samples = downsample_block(
                            hidden_states=sample,
                            temb=emb,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                            cross_attention_kwargs=cross_attention_kwargs,
                        )
                    else:
                        sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                    #---------------------------------
                    # torch.save(sample, f'{timestep.item()}_step/down_{i}.pt')
                    #----------------------------------
                    down_block_res_samples += res_samples

                if down_block_additional_residuals is not None:
                    new_down_block_res_samples = ()

                    for down_block_res_sample, down_block_additional_residual in zip(
                        down_block_res_samples, down_block_additional_residuals
                    ):
                        down_block_res_sample = down_block_res_sample + down_block_additional_residual
                        new_down_block_res_samples += (down_block_res_sample,)

                    down_block_res_samples = new_down_block_res_samples

                # 4. mid
                if self.mid_block is not None:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                    # torch.save(sample, f'{timestep.item()}_step/mid.pt')
                if mid_block_additional_residual is not None:
                    sample = sample + mid_block_additional_residual
                # 5. up
                for i, upsample_block in enumerate(self.up_blocks):
                    is_final_block = i == len(self.up_blocks) - 1

                    res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                    down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                    # if we have not reached the final block and need to forward the
                    # upsample size, we do it here
                    if not is_final_block and forward_upsample_size:
                        upsample_size = down_block_res_samples[-1].shape[2:]

                    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                        )
                    #----------------------------
                    # torch.save(sample, f'{timestep.item()}_step/up_{i}.pt')
                    #----------------------------
                # 6. post-process
                if self.conv_norm_out:
                    sample = self.conv_norm_out(sample)
                    sample = self.conv_act(sample)
                sample = self.conv_out(sample)

                if not return_dict:
                    return (sample,)

                return UNet2DConditionOutput(sample=sample)
        return forward
    if model.__class__.__name__ == 'UNet2DConditionModel':
        model.forward = normal_forward(model)

def refister_time(unet, t):
    setattr(unet, 'order', t)



def register_controlnet_pipeline2(pipe):
    def new_call(self):
        @torch.no_grad()
        # @replace_example_docstring(EXAMPLE_DOC_STRING)
        def call(
            prompt: Union[str, List[str]] = None,
            image: Union[
                torch.FloatTensor,
                PIL.Image.Image,
                np.ndarray,
                List[torch.FloatTensor],
                List[PIL.Image.Image],
                List[np.ndarray],
            ] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            guess_mode: bool = False,
        ):
            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                image,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                controlnet_conditioning_scale,
            )

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

            if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

            global_pool_conditions = (
                controlnet.config.global_pool_conditions
                if isinstance(controlnet, ControlNetModel)
                else controlnet.nets[0].config.global_pool_conditions
            )
            guess_mode = guess_mode or global_pool_conditions

            # 3. Encode input prompt
            text_encoder_lora_scale = (
                cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
            )
            prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
            )

            # 4. Prepare image
            if isinstance(controlnet, ControlNetModel):
                image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = image.shape[-2:]
            elif isinstance(controlnet, MultiControlNetModel):
                images = []

                for image_ in image:
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    images.append(image_)

                image = images
                height, width = image[0].shape[-2:]
            else:
                assert False

            # 5. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 6. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            self.init_latent = latents.detach().clone()
            # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 8. Denoising loop
            #-------------------------------------------------------------
            all_steps = len(self.scheduler.timesteps)
            curr_span = 1
            curr_step = 0

            # st = time.time()
            idx = 1
            keytime = [0,1,2,3,5,10,15,25,35,50]
            
            while curr_step<all_steps:
                # torch.cuda.empty_cache()
                # print(curr_step)
                refister_time(self.unet, curr_step)

                merge_span = curr_span
                if merge_span>0:
                    time_ls = []
                    for i in range(curr_step, curr_step+merge_span):
                        if i<all_steps:
                            time_ls.append(self.scheduler.timesteps[i])
                        else:
                            break
                    # torch.cuda.empty_cache()

                    ##--------------------------------
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, time_ls[0])
                    
                    if curr_step in [0,1,2,3,5,10,15,25,35]:
                        # controlnet(s) inference
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds

                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            control_model_input,
                            time_ls[0],
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=image,
                            conditioning_scale=controlnet_conditioning_scale,
                            guess_mode=guess_mode,
                            return_dict=False,
                        )


                        #----------------------save controlnet feature-------------------------
                        #useless, shoule delete
                        # setattr(self, 'downres_samples', deepcopy(down_block_res_samples))
                        # setattr(self, 'midres_sample', mid_block_res_sample.detach().clone())
                        #-----------------------save controlnet feature------------------------
                    else:
                        down_block_res_samples = None #self.downres_samples
                        mid_block_res_sample = None #self.midres_sample
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        time_ls,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1

                    if isinstance(time_ls, list):
                        step_span = len(time_ls)
                        bs = noise_pred.shape[0]
                        bs_perstep = bs//step_span

                        denoised_latent = latents
                        for i, timestep in enumerate(time_ls):
                            curr_noise = noise_pred[i*bs_perstep:(i+1)*bs_perstep]
                            denoised_latent = self.scheduler.step(curr_noise, timestep, denoised_latent, **extra_step_kwargs, return_dict=False)[0]
                        
                        latents = denoised_latent
                    ##----------------------------------------
                curr_step += curr_span
                idx += 1
                if curr_step<all_steps:
                    curr_span = keytime[idx] - keytime[idx-1]
            
            # for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):

            #-------------------------------------------------------------
                

            # If we do sequential model offloading, let's offload unet and controlnet
            # manually for max memory savings
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                self.unet.to("cpu")
                self.controlnet.to("cpu")
                torch.cuda.empty_cache()

            if not output_type == "latent":
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            # Offload last model to CPU
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                self.final_offload_hook.offload()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        return call
    pipe.call = new_call(pipe)

@torch.no_grad()
def multistep_pre(self, noise_pred, t, x):
    step_span = len(t)
    bs = noise_pred.shape[0]
    bs_perstep = bs//step_span

    denoised_latent = x
    for i, timestep in enumerate(t):
        curr_noise = noise_pred[i*bs_perstep:(i+1)*bs_perstep]
        denoised_latent = self.scheduler.step(curr_noise, timestep, denoised_latent)['prev_sample']
    return denoised_latent

def register_t2v(model):
    def new_back(self):
        def backward_loop(
        latents,
        timesteps,
        prompt_embeds,
        guidance_scale,
        callback,
        callback_steps,
        num_warmup_steps,
        extra_step_kwargs,
        cross_attention_kwargs=None,):
            do_classifier_free_guidance = guidance_scale > 1.0
            num_steps = (len(timesteps) - num_warmup_steps) // self.scheduler.order
            import time
            if num_steps<10:
                with self.progress_bar(total=num_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        setattr(self.unet, 'order', i)
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                        # call the callback, if provided
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and i % callback_steps == 0:
                                step_idx = i // getattr(self.scheduler, "order", 1)
                                callback(step_idx, t, latents)

            else:
                all_timesteps = len(timesteps)
                curr_step = 0
   
                while curr_step<all_timesteps:
                    refister_time(self.unet, curr_step)

                    time_ls = []
                    time_ls.append(timesteps[curr_step])
                    curr_step += 1
                    cond = curr_step in [0,1,2,3,5,10,15,25,35]
                    
                    while (not cond) and (curr_step<all_timesteps):
                        time_ls.append(timesteps[curr_step])
                        curr_step += 1
                        cond = curr_step in [0,1,2,3,5,10,15,25,35]
                    
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        time_ls,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = multistep_pre(self, noise_pred, time_ls, latents)
             
            return latents.clone().detach()
        return backward_loop
    model.backward_loop = new_back(model)
    