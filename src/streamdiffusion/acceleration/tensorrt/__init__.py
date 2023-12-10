import gc
import os
from typing import Literal

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from polygraphy import cuda

from ...pipeline import StreamDiffusion
from .builder import EngineBuilder, create_onnx_path
from .engine import AutoencoderKLEngine, UNet2DConditionModelEngine
from .models import VAE, UNet, VAEEncoder


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        return retrieve_latents(self.vae.encode(x), torch.Generator())


def accelerate_with_tensorrt(
    stream: StreamDiffusion,
    engine_dir: str,
    max_batch_size: int = 2,
    min_batch_size: int = 1,
    use_cuda_graph: bool = False,
    mode: Literal["img2img", "txt2img"] = "img2img",
    engine_build_options: dict = {},
):
    if (
        "opt_batch_size" not in engine_build_options
        or engine_build_options["opt_batch_size"] is None
    ):
        engine_build_options["opt_batch_size"] = max_batch_size
    text_encoder = stream.text_encoder
    unet = stream.unet
    vae = stream.vae

    del stream.unet, stream.vae, stream.pipe.unet, stream.pipe.vae

    vae_config = vae.config
    vae_dtype = vae.dtype

    unet.to(torch.device("cpu"))
    vae.to(torch.device("cpu"))

    gc.collect()
    torch.cuda.empty_cache()

    onnx_dir = os.path.join(engine_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    unet_engine_path = f"{engine_dir}/unet.engine"
    vae_encoder_engine_path = f"{engine_dir}/vae_encoder.engine"
    vae_decoder_engine_path = f"{engine_dir}/vae_decoder.engine"

    unet_model = UNet(
        fp16=True,
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        embedding_dim=text_encoder.config.hidden_size,
        unet_dim=unet.config.in_channels,
    )

    if mode == "img2img":
        vae_decoder_model = VAE(
            device=stream.device, max_batch_size=max_batch_size, min_batch_size=1
        )
        vae_encoder_model = VAEEncoder(
            device=stream.device, max_batch_size=max_batch_size, min_batch_size=1
        )
    elif mode == "txt2img":
        vae_decoder_model = VAE(
            device=stream.device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
        )
        vae_encoder_model = VAEEncoder(
            device=stream.device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
        )
    else:
        raise ValueError(
            f"mode should be one of ['img2img', 'txt2img'], but got {mode}"
        )

    if not os.path.exists(unet_engine_path):
        unet = unet.to(stream.device, dtype=torch.float16)
        builder = EngineBuilder(unet_model, unet, device=stream.device)
        del unet
        builder.build(
            create_onnx_path("unet", onnx_dir, opt=False),
            create_onnx_path("unet", onnx_dir, opt=True),
            unet_engine_path,
            **engine_build_options,
        )
    else:
        del unet

    if not os.path.exists(vae_decoder_engine_path):
        vae.forward = vae.decode
        vae = vae.to(stream.device)
        builder = EngineBuilder(vae_decoder_model, vae, device=stream.device)
        builder.build(
            create_onnx_path("vae_decoder", onnx_dir, opt=False),
            create_onnx_path("vae_decoder", onnx_dir, opt=True),
            vae_decoder_engine_path,
            **engine_build_options,
        )

    if not os.path.exists(vae_encoder_engine_path):
        vae_encoder = TorchVAEEncoder(vae).to(stream.device)
        builder = EngineBuilder(vae_encoder_model, vae_encoder, device=stream.device)
        builder.build(
            create_onnx_path("vae_encoder", onnx_dir, opt=False),
            create_onnx_path("vae_encoder", onnx_dir, opt=True),
            vae_encoder_engine_path,
            **engine_build_options,
        )
        del vae_encoder
        gc.collect()
        torch.cuda.empty_cache()

    del vae

    cuda_steram = cuda.Stream()

    stream.unet = UNet2DConditionModelEngine(
        unet_engine_path, cuda_steram, use_cuda_graph=use_cuda_graph
    )
    stream.vae = AutoencoderKLEngine(
        vae_encoder_engine_path,
        vae_decoder_engine_path,
        cuda_steram,
        stream.pipe.vae_scale_factor,
        use_cuda_graph=use_cuda_graph,
    )
    setattr(stream.vae, "config", vae_config)
    setattr(stream.vae, "dtype", vae_dtype)

    gc.collect()
    torch.cuda.empty_cache()

    return stream
