# StreamDiffusion

**[StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation](https://arxiv.org/abs/2312.12491)**
</br>
[Akio Kodaira*](https://www.linkedin.com/feed/),
[Chenfeng Xu*](https://www.chenfengx.com/),
[Toshiki Hazama*](xxx),
[Takanori Yoshimoto](xxx),
[Kohei Ohno](https://www.linkedin.com/in/kohei--ohno/),
[Shogo Mitsuhori](xxx),
[Soichi Suganoo](xxx),
[Hanying Cho](xxx),
[Zhijian Liu](https://scholar.google.com/citations?hl=en&user=3coYSTUAAAAJ),
[Kurt Keutzer](https://scholar.google.com/citations?hl=en&user=ID9QePIAAAAJ),
(*Corresponding Author)


[![arXiv](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)](https://arxiv.org/abs/2312.12491)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/papers/2312.12491)


## Installation

### Step0: Make environment
```
conda create -n stream-diffusion python=3.10
conda activate stream-diffusion
```
OR
```
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux
source .venv/bin/activate
```

### Step1: Install Torch
Select the appropriate version for your system.

CUDA 11.8
```
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu118
```
CUDA 12.1
```
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
```
details: https://pytorch.org/

### Step2: Install StreamDiffusion

#### For User
Install StreamDiffusion
```
pip install git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=stream-diffusion
```
Install tensorrt extension
```
python -m streamdiffusion.tools.install-tensorrt
```

#### For Developer
```
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
python setup.py develop easy_install stream-diffusion[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
```

## Quick Start

You can try StreamDiffusion in [`examples`](./examples) directory.

| ![画像1](./assets/demo_02.gif) | ![画像2](./assets/demo_03.gif) |
|:--------------------:|:--------------------:|
| ![画像3](./assets/demo_04.gif) | ![画像4](./assets/demo_05.gif) |

## Usage

```python
from typing import Literal, Optional
from PIL import Image
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from tqdm import tqdm

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import pil2tensor, postprocess_image


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def run(
    prompt: str = "1girl with dog hair, thick frame glasses",
    warmup: int = 10,
    iterations: int = 50,
    lcm_lora: bool = True,
    tiny_vae: bool = True,
    acceleration: Optional[Literal["none", "xformers", "tensorrt"]] = "xformers",
):
    # Load Stable Diffusion pipeline
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
        "KBlueLeaf/kohaku-v2.1"
    ).to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

    # Wrap the pipeline in StreamDiffusion
    stream = StreamDiffusion(
        pipe,
        [32, 45],
        torch_dtype=torch.float16,
    )

    # Load LCM LoRA
    if lcm_lora:
        stream.load_lcm_lora()
        stream.fuse_lora()

    # Load Tiny VAE
    if tiny_vae:
        stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
            device=pipe.device, dtype=pipe.dtype
        )

    # Enable acceleration
    if acceleration == "xformers":
        pipe.enable_xformers_memory_efficient_attention()
    elif acceleration == "tensorrt":
        from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

        stream = accelerate_with_tensorrt(
            stream,
            "engines",
            max_batch_size=2,
            engine_build_options={"build_static_batch": True},
        )

    # Prepare the stream
    stream.prepare(
        prompt,
        num_inference_steps=50,
    )

    # Prepare the input tensor
    image = Image.open("assets/img2img_example.png").convert("RGB").resize((512, 512))
    input_tensor = pil2tensor(image)

    # Warmup
    for _ in range(warmup):
        stream(
            input_tensor.detach().clone().to(device=stream.device, dtype=stream.dtype)
        )


    # Run the stream
    results = []
    for _ in tqdm(range(iterations)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        x_output = stream(
            input_tensor.detach().clone().to(device=stream.device, dtype=stream.dtype)
        )
        image = postprocess_image(x_output, output_type="pil")[0]
        end.record()

        torch.cuda.synchronize()
        results.append(start.elapsed_time(end))

    print(f"Average time: {sum(results) / len(results)}ms")
    print(f"Average FPS: {1000 / (sum(results) / len(results))}")


if __name__ == "__main__":
    run()
```

# Acknowledgements
</br>
[Aki](https://github.com/cumulo-autumn/),
[Ararat](https://github.com/AttaQ/),
[Chenfeng Xu](https://github.com/chenfengxu714/),
[ddPn08](https://github.com/ddPn08/),
[kizamimi](https://github.com/kizamimi/),
[ramune](https://github.com/YN35/),
[teftef](https://github.com/teftef6220/),
[Tonimono](https://github.com/mili-inch/),
[Verb](https://github.com/discus0434),

(*alphabetical order)

</br>
The video and image demos in this github repository were generated using [kohakuV2](https://civitai.com/models/136268/kohaku-v2). Thanks to Kohaku BlueLeaf ([@KBlueleaf](https://twitter.com/KBlueleaf)) for providing the model.

Can download model in [Civitai](https://civitai.com/models/136268/kohaku-v2) and [HuggingFace](https://huggingface.co/KBlueLeaf/kohaku-v2.1/tree/main).