# StreamDiffusion

![](https://img.shields.io/badge/%F0%9F%A4%97%20-Hugging%20Face%20Spaces-blue?style=for-the-badge)
![](https://img.shields.io/badge/Open%20in%20Colab-blue?style=for-the-badge&logo=googlecolab&labelColor=5c5c5c)

## Installation

### Step0: Make environment
```
conda create -n stream-diffusion python=3.10
conda activate stream-diffusion
```
OR
```
python -m venv .venv
```

### Step1: Install Torch
Select the appropriate version for your system.

CUDA 11.1
```
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu118
```
CUDA 12.1
```
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu121
```
details: https://pytorch.org/

### Step2: Install StreamDiffusion
```
pip install git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=stream-diffusion
```
OR
```
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
python -m pip install .
```
OR [if you want to use tensorrt]
```
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
pip install .[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
```
OR [if you are a developer]
```
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
python setup.py develop easy_install stream-diffusion[dev]
python -m streamdiffusion.tools.install-tensorrt
```

## Usage
```python
import io
from typing import *

import fire
import PIL.Image
import requests
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from tqdm import tqdm

from streamdiffusion import StreamDiffusion


def download_image(url: str):
    response = requests.get(url)
    image = PIL.Image.open(io.BytesIO(response.content))
    return image


def run(
    wamup: int = 10,
    iterations: int = 50,
    prompt: str = "Girl with panda ears wearing a hood",
    lcm_lora: bool = True,
    tiny_vae: bool = True,
    acceleration: Optional[Literal["xformers", "sfast", "tensorrt"]] = None,
    device_ids: Optional[List[int]] = None,
):
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file("./model.safetensors").to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    stream = StreamDiffusion(
        pipe,
        [32, 45],
        torch_dtype=torch.float16,
        width=512,
        height=512,
    )

    if lcm_lora:
        stream.load_lcm_lora()
        stream.fuse_lora()

    if tiny_vae:
        stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

    if device_ids is not None:
        stream.unet = torch.nn.DataParallel(stream.unet, device_ids=device_ids)

    if acceleration == "xformers":
        pipe.enable_xformers_memory_efficient_attention()
    elif acceleration == "tensorrt":
        from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

        stream = accelerate_with_tensorrt(stream, "engines", max_batch_size=2)
    elif acceleration == "sfast":
        from streamdiffusion.acceleration.sfast import accelerate_with_stable_fast

        stream = accelerate_with_stable_fast(stream)

    stream.prepare(
        prompt,
        num_inference_steps=50,
    )

    image = download_image("https://github.com/ddpn08.png")

    # warmup
    for _ in range(wamup):
        stream(image)

    results = []

    for _ in tqdm(range(iterations)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        x_output = stream(image)
        stream.image_processor.postprocess(x_output, output_type="pil")
        end.record()

        torch.cuda.synchronize()
        results.append(start.elapsed_time(end))

    print(f"Average time: {sum(results) / len(results)}ms")
    print(f"Average FPS: {1000 / (sum(results) / len(results))}")


if __name__ == "__main__":
    fire.Fire(run)

``````