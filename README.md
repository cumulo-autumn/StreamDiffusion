# StreamDiffusion

![](https://img.shields.io/badge/%F0%9F%A4%97%20-Hugging%20Face%20Spaces-blue?style=for-the-badge)
![](https://img.shields.io/badge/Open%20in%20Colab-blue?style=for-the-badge&logo=googlecolab&labelColor=5c5c5c)

## Installation

### Step0: Make conda environment
```
conda create -n stream-diffusion python=3.10
```

### Step1: Install Torch
Select the appropriate version for your system.

CUDA 11.1
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
CUDA 12.1
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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
OR (if you want to use tensorrt)
```
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
pip install .[tensorrt]
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
from diffusers import AutoencoderTiny, LCMScheduler, StableDiffusionPipeline
from tqdm import tqdm

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import pil2tensor, postprocess_image


def download_image(url: str):
    response = requests.get(url)
    image = PIL.Image.open(io.BytesIO(response.content))
    return image


def run(wamup: int = 10, iterations: int = 50):
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file("./models/model.safetensors").to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()
    pipe.enable_xformers_memory_efficient_attention()

    stream = StreamDiffusion(
        pipe,
        [32, 45],
        torch_dtype=torch.float16,
    )

    stream.prepare(
        "Girl with panda ears wearing a hood",
        num_inference_steps=50,
        generator=torch.manual_seed(2),
    )

    image = download_image("https://github.com/ddpn08.png").resize((512, 512))
    input_tensor = pil2tensor(image)

    # warmup
    for _ in range(wamup):
        stream(input_tensor.detach().clone().to(device=stream.device, dtype=stream.dtype))

    results = []

    for _ in tqdm(range(iterations)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        x_output = stream(input_tensor.detach().clone().to(device=stream.device, dtype=stream.dtype))
        postprocess_image(x_output, output_type="pil")[0]
        end.record()

        torch.cuda.synchronize()
        results.append(start.elapsed_time(end))

    print(f"Average time: {sum(results) / len(results)}ms")
    print(f"Average FPS: {1000 / (sum(results) / len(results))}")


if __name__ == "__main__":
    fire.Fire(run)
``````