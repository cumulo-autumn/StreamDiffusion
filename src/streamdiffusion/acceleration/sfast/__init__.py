from typing import Optional

from sfast.compilers.stable_diffusion_pipeline_compiler import CompilationConfig, compile

from ...pipeline import StreamDiffusion


def accelerate_with_stable_fast(
    stream: StreamDiffusion,
    config: Optional[CompilationConfig] = None,
):
    if config is None:
        config = CompilationConfig.Default()
        # xformers and Triton are suggested for achieving best performance.
        try:
            import xformers

            config.enable_xformers = True
        except ImportError:
            print("xformers not installed, skip")
        try:
            import triton

            config.enable_triton = True
        except ImportError:
            print("Triton not installed, skip")
        # CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
        config.enable_cuda_graph = True
    stream.pipe = compile(stream.pipe, config)
    stream.unet = stream.pipe.unet
    stream.vae = stream.pipe.vae
    stream.text_encoder = stream.pipe.text_encoder
    return stream
