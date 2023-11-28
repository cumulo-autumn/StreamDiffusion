import time
import torch
from diffusers import LCMScheduler, AutoencoderTiny, StableDiffusionPipeline
from TensorRT.trt_net import TrtUnet
from utils.image_process import postprocess
import config

class StreamDiffusion:
    def __init__(self, config, device, model_path, lora_paths, prompt, t_index_list, use_tiny_vae = True, use_xformers = False, use_trt = False, generator = None, num_inference_steps = 50):
        torch.set_grad_enabled(False)

        self.config = config

        self.device = device

        self.batch_size = len(t_index_list)

        # prepare pipeline and vae
        pipe = StableDiffusionPipeline.from_single_file(model_path).to(self.device)
        for lora_path in lora_paths:
            pipe.load_lora_weights(lora_path)
        if use_xformers:
            pipe.enable_xformers_memory_efficient_attention()
        pipe.to(torch_device=self.device, torch_dtype=torch.float16)
        self.config = config

        self.use_tiny_vae = use_tiny_vae
        if self.use_tiny_vae:
            self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(self.device)
        else:
            self.vae = pipe.vae

        self.latent_height = config.height // self.vae.spatial_scale_factor
        self.latent_width = config.width // self.vae.spatial_scale_factor

        # initialize x_t_latent (it can be any random tensor)
        if len(t_index_list) > 1:
            self.x_t_latent_buffer = torch.zeros((self.batch_size-1, 4, self.latent_height, self.latent_width), dtype=torch.float16).to(self.device)
        else:
            self.x_t_latent_buffer = None

        self.additional_embeds = pipe.encode_prompt(prompt=prompt,device=self.device, num_images_per_prompt=1,do_classifier_free_guidance=False)
        self.prompt_embeds = self.additional_embeds[0].repeat(self.batch_size, 1, 1)

        self.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        self.unet = pipe.unet

        self.t_list = t_index_list

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        self.sub_timesteps_tensor = torch.tensor(self.sub_timesteps, dtype=torch.long, device=self.device)

        if generator is not None:
            self.init_noise = torch.randn((self.batch_size, 4, self.latent_height, self.latent_width), generator=generator, dtype=torch.float16).to(self.device)
        else:
            self.init_noise = torch.randn((self.batch_size, 4, self.latent_height, self.latent_width), dtype=torch.float16).to(self.device)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)
        self.c_skip = torch.stack(c_skip_list).view(self.batch_size,1,1,1).type(torch.float16).to(self.device)
        self.c_out = torch.stack(c_out_list).view(self.batch_size,1,1,1).type(torch.float16).to(self.device)
        
        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        self.alpha_prod_t_sqrt = torch.stack(alpha_prod_t_sqrt_list).view(self.batch_size,1,1,1).type(torch.float16).to(self.device)
        self.beta_prod_t_sqrt = torch.stack(beta_prod_t_sqrt_list).view(self.batch_size,1,1,1).type(torch.float16).to(self.device)

        self.use_trt = use_trt
        if use_trt:
            self.trt_unet = TrtUnet(config.trt_model_path)
            self.trt_unet.activate()

    def add_noise(self, original_samples, noise, t_index):
        noisy_samples = self.alpha_prod_t_sqrt[t_index] * original_samples + self.beta_prod_t_sqrt[t_index] * noise
        return noisy_samples
    
    def scheduler_step_batch(self, model_pred_batch, x_t_latent_batch):
        F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch)/self.alpha_prod_t_sqrt
        denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        return denoised_batch

    def lcm_step(
        self, 
        x_t_latent: torch.FloatTensor,
    ):
        if self.use_trt:
            model_pred = self.trt_unet.forward(x_t_latent, self.sub_timesteps_tensor, self.prompt_embeds)
        else:
            model_pred = self.unet(
                x_t_latent,
                self.sub_timesteps_tensor,
                encoder_hidden_states=self.prompt_embeds,
            )[0]

        # compute the previous noisy sample x_t -> x_t-1
        if self.batch_size > 1:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent)
        else:
            x_t_next, x_0_predicted = self.scheduler.step(model_pred, self.sub_timesteps_tensor, x_t_latent, return_dict=False)
            denoised_batch = x_0_predicted
        
        return denoised_batch, model_pred

    def encode_image(self, image_tensors):
        start_time = time.time()
        image_tensors = image_tensors.to(self.device)
        if self.use_tiny_vae:
            img_latent = self.vae.encode(image_tensors).latents * self.vae.config.scaling_factor
        else:
            img_latent = self.vae.encode(image_tensors).latent_dist.sample() * self.vae.config.scaling_factor
        # print("encode_time: ", time.time() - start_time)
        start_time = time.time()
        x_t_latent = self.add_noise(img_latent, self.init_noise[0], 0)
        # print("adding_noise_time: ", time.time() - start_time)
        return x_t_latent

    def decode_image(self, x_0_pred_out):
        start_time = time.time()
        # output_latent = self.vae.decode(x_0_pred_out / self.vae.config.scaling_factor, return_dict=False)[0].detach().clone()
        output_latent = self.vae.decode(x_0_pred_out / self.vae.config.scaling_factor, return_dict=False)[0]
        # print("decode_time: ", time.time() - start_time)
        return output_latent

    def predict_x0_batch(self, x_t_latent):
        # start_time = time.time()
        prev_latent_batch = self.x_t_latent_buffer
        if self.batch_size > 1:
            x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)

        x_0_pred_batch, model_pred = self.lcm_step(x_t_latent)
        # print("lcm_step_time: ", time.time() - start_time)
        # start_time = time.time()
        if prev_latent_batch is not None:
            x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
            if self.config.is_drawing:
                self.x_t_latent_buffer = self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1] + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
            else:
                self.x_t_latent_buffer = self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
        else:
            x_0_pred_out = x_0_pred_batch
            self.x_t_latent_buffer = None
        # print("for_loop_time: ", time.time() - start_time)
        # print("lcm_process_time: ", time.time() - start_time)
        return x_0_pred_out
    


class StreamLCM_Unet:
    def __init__(self, config, device, model_path, lora_paths, prompt, t_index_list, use_tiny_vae = True, use_xformers = False, use_trt = False, generator = None, num_inference_steps = 50):
        torch.set_grad_enabled(False)

        self.config = config

        self.device = device

        self.batch_size = len(t_index_list)



        # prepare pipeline
        pipe = StableDiffusionPipeline.from_single_file(model_path).to(self.device)
        for lora_path in lora_paths:
            pipe.load_lora_weights(lora_path)
        if use_xformers:
            pipe.enable_xformers_memory_efficient_attention()
        pipe.to(torch_device=self.device, torch_dtype=torch.float16)
        self.config = config

        self.latent_height = config.height // pipe.vae.spatial_scale_factor
        self.latent_width = config.width // pipe.vae.spatial_scale_factor

        # initialize x_t_latent (it can be any random tensor)
        if len(t_index_list) > 1:
            self.x_t_latent_buffer = torch.zeros((self.batch_size-1, 4, self.latent_height, self.latent_width), dtype=torch.float16).to(self.device)
        else:
            self.x_t_latent_buffer = None

        self.additional_embeds = pipe.encode_prompt(prompt=prompt,device=self.device, num_images_per_prompt=1,do_classifier_free_guidance=False)
        self.prompt_embeds = self.additional_embeds[0].repeat(self.batch_size, 1, 1)

        self.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        self.unet = pipe.unet
        self.t_list = t_index_list

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        self.sub_timesteps_tensor = torch.tensor(self.sub_timesteps, dtype=torch.long, device=self.device)

        if generator is not None:
            self.init_noise = torch.randn((self.batch_size, 4, self.latent_height, self.latent_width), generator=generator, dtype=torch.float16).to(self.device)
        else:
            self.init_noise = torch.randn((self.batch_size, 4, self.latent_height, self.latent_width), dtype=torch.float16).to(self.device)


        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)
        self.c_skip = torch.stack(c_skip_list).view(self.batch_size,1,1,1).type(torch.float16).to(self.device)
        self.c_out = torch.stack(c_out_list).view(self.batch_size,1,1,1).type(torch.float16).to(self.device)
        
        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        self.alpha_prod_t_sqrt = torch.stack(alpha_prod_t_sqrt_list).view(self.batch_size,1,1,1).type(torch.float16).to(self.device)
        self.beta_prod_t_sqrt = torch.stack(beta_prod_t_sqrt_list).view(self.batch_size,1,1,1).type(torch.float16).to(self.device)

        self.use_trt = use_trt
        if use_trt:
            self.trt_unet = TrtUnet(config.trt_model_path)
            self.trt_unet.activate()

    def add_noise(self, original_samples, noise, t_index):
        noisy_samples = self.alpha_prod_t_sqrt[t_index] * original_samples + self.beta_prod_t_sqrt[t_index] * noise
        return noisy_samples
    
    def scheduler_step_batch(self, model_pred_batch, x_t_latent_batch):
        F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch)/self.alpha_prod_t_sqrt
        denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        return denoised_batch

    def lcm_step(
        self, 
        x_t_latent: torch.FloatTensor,
    ):
        if self.use_trt:
            model_pred = self.trt_unet.forward(x_t_latent, self.sub_timesteps_tensor, self.prompt_embeds)
        else:
            model_pred = self.unet(
                x_t_latent,
                self.sub_timesteps_tensor,
                encoder_hidden_states=self.prompt_embed,
            )[0]

        # compute the previous noisy sample x_t -> x_t-1
        if self.batch_size > 1:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent)
        else:
            x_t_next, x_0_predicted = self.scheduler.step(model_pred, self.sub_timesteps_tensor, x_t_latent, return_dict=False)
            denoised_batch = x_0_predicted
        
        return denoised_batch, model_pred

    def lcm_sdxl_step(
        self, 
        x_t_latent: torch.FloatTensor,
    ):
        # SDXLは若干仕様が違うので、今後使う時のために別途定義
        raise NotImplementedError

    def predict_x0_batch(self, x_t_latent):
        # start_time = time.time()
        prev_latent_batch = self.x_t_latent_buffer
        if self.batch_size > 1:
            x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)

        x_0_pred_batch, model_pred = self.lcm_step(x_t_latent)
        # print("lcm_step_time: ", time.time() - start_time)
        # start_time = time.time()
        if prev_latent_batch is not None:
            x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
            if self.config.is_drawing:
                self.x_t_latent_buffer = self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1] + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
            else:
                self.x_t_latent_buffer = self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
        else:
            x_0_pred_out = x_0_pred_batch
            self.x_t_latent_buffer = None
        # print("for_loop_time: ", time.time() - start_time)
        # print("lcm_process_time: ", time.time() - start_time)
        return x_0_pred_out
    


class StreamLCM_VAE:
    def __init__(self, device = "cuda", generator = None, num_inference_steps = 50):
        torch.set_grad_enabled(False)
        self.device = device
        self.batch_size = len(config.t_list)
        self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(self.device)

        self.latent_height = self.height // pipe.vae.spatial_scale_factor
        self.latent_width = self.width // pipe.vae.spatial_scale_factor

        if generator is not None:
            self.init_noise = torch.randn((self.batch_size, 4, self.latent_height, self.latent_width), generator=generator, dtype=torch.float16).to(self.device)
        else:
            self.init_noise = torch.randn((self.batch_size, 4, self.latent_height, self.latent_width), dtype=torch.float16).to(self.device)

        #### temporary (get noise scheduler values) ####
        pipe = StableDiffusionPipeline.from_single_file(config.model_path).to(self.device)
        for lora_path in config.lora_paths:
            pipe.load_lora_weights(lora_path)
        if config.use_xformers:
            pipe.enable_xformers_memory_efficient_attention()
        pipe.to(torch_device=self.device, torch_dtype=torch.float16)

        scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = scheduler.timesteps.to(self.device)
        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        sub_timesteps = []
        for t in config.t_list:
            sub_timesteps.append(timesteps[t])
        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in sub_timesteps:
            alpha_prod_t_sqrt = scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        ###############################################

        self.alpha_prod_t_sqrt = torch.stack(alpha_prod_t_sqrt_list).view(self.batch_size,1,1,1).type(torch.float16).to(self.device)
        self.beta_prod_t_sqrt = torch.stack(beta_prod_t_sqrt_list).view(self.batch_size,1,1,1).type(torch.float16).to(self.device)


    def add_noise(self, original_samples, noise, t_index):
        noisy_samples = self.alpha_prod_t_sqrt[t_index] * original_samples + self.beta_prod_t_sqrt[t_index] * noise
        return noisy_samples

    def encode_image(self, image_tensors):
        start_time = time.time()
        image_tensors = image_tensors.to(self.device)
        img_latent = self.vae.encode(image_tensors).latents * self.vae.config.scaling_factor
        # print("encode_time: ", time.time() - start_time)
        start_time = time.time()
        x_t_latent = self.add_noise(img_latent, self.init_noise[0], 0)
        # print("adding_noise_time: ", time.time() - start_time)
        return x_t_latent

    def decode_image(self, x_0_pred_out):
        start_time = time.time()
        output_latent = self.vae.decode(x_0_pred_out / self.vae.config.scaling_factor, return_dict=False)[0].detach().clone()
        # print("decode_time: ", time.time() - start_time)
        return output_latent