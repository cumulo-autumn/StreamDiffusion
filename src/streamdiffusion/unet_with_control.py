import torch
from diffusers import ControlNetModel, UNet2DConditionModel


class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel, controlnets: list[ControlNetModel], controlnet_scales: list[float]):
        super().__init__()
        self.unet = unet.to(unet.device, dtype=unet.dtype)
        self.controlnets = [controlnet.to(unet.device, dtype=unet.dtype) for controlnet in controlnets]
        self.controlnet_scales = controlnet_scales

    def forward(self, sample, timestep, encoder_hidden_states, controlnet_images) -> torch.Tensor:
        for i in range(len(self.controlnets)):
            down_samples, mid_sample = self.controlnets[i](
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_images[i],
                guess_mode=False,
                return_dict=False,
            )

            down_samples = [down_sample * self.controlnet_scales[i] for down_sample in down_samples]
            mid_sample *= self.controlnet_scales[i]

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )
        return noise_pred
