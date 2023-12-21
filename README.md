# StreamDiffusion

[English](./README.md) | [日本語](./README-ja.md)

<p align="center">
  <img src="./assets/demo_07.gif" width=90%>
  <img src="./assets/demo_09.gif" width=90%>
</p>

# StreamDiffusion: A Pipeline-Level Solution for Real-Time Interactive Generation

**Authors:** [Akio Kodaira](https://www.linkedin.com/feed/), [Chenfeng Xu](https://www.chenfengx.com/), Toshiki Hazama, Takanori Yoshimoto, [Kohei Ohno](https://www.linkedin.com/in/kohei--ohno/), [Shogo Mitsuhori](https://me.ddpn.world/), [Soichi Sugano](https://twitter.com/toni_nimono), Hanying Cho, [Zhijian Liu](https://zhijianliu.com/), [Kurt Keutzer](https://scholar.google.com/citations?hl=en&user=ID9QePIAAAAJ)

StreamDiffusion is an innovative diffusion pipeline designed for real-time interactive generation. It introduces significant performance enhancements to current diffusion-based image generation techniques.


[![arXiv](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)](https://arxiv.org/abs/2312.12491)
[![Hugging Face Papers](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-papers-yellow)](https://huggingface.co/papers/2312.12491)

We sincerely thank [Taku Fujimoto](https://twitter.com/AttaQjp) and [Radamés Ajna](https://twitter.com/radamar) and Huggingface team for their invaluable feedback, courteous support, and insightful discussions.

## Key Features

1. **Stream Batch** - [Learn More](#stream-batching-link)
   - Streamlined data processing through efficient batch operations.

2. **Residual Classifier-Free Guidance** - [Learn More](#residual-classifier-free-guidance-link)
   - Improved guidance mechanism that minimizes computational redundancy.

3. **Stochastic Similarity Filter** - [Learn More](#stochastic-similarity-filtering-link)
   - Improves GPU utilization efficiency through advanced filtering techniques.

4. **IO Queues** - [Learn More](#io-queues-link)
   - Efficiently manages input and output operations for smoother execution.

5. **Pre-computation for KV-Caches** - [Learn More](#pre-computation-for-kv-caches-link)
   - Optimizes caching strategies for accelerated processing.

6. **Model Acceleration Tools**
   - Utilizes various tools for model optimization and performance boost.

Feel free to explore each feature by following the provided links to learn more about StreamDiffusion's capabilities. If you find it helpful, please consider citing our work:


```bash
@article{kodaira2023streamdiffusion,
      title={StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation},
      author={Akio Kodaira and Chenfeng Xu and Toshiki Hazama and Takanori Yoshimoto and Kohei Ohno and Shogo Mitsuhori and Soichi Sugano and Hanying Cho and Zhijian Liu and Kurt Keutzer},
      year={2023},
      eprint={2312.12491},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Installation

### Step0: Make environment

```bash
conda create -n streamdiffusion python=3.10
conda activate streamdiffusion
```

OR

```cmd
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux
source .venv/bin/activate
```

### Step1: Install Torch
Select the appropriate version for your system.

CUDA 11.8

```bash
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu118
```

CUDA 12.1

```bash
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
```
details: https://pytorch.org/

### Step2: Install StreamDiffusion

#### For User
Install StreamDiffusion

```bash
pip install git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=streamdiffusion
```

Install TensorRT extension

```bash
python -m streamdiffusion.tools.install-tensorrt
```

#### For Developer

```bash
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
python setup.py develop easy_install streamdiffusion[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
```

## Docker Installation (TensorRT Ready)

```bash
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
cd StreamDiffusion
docker build -t stream-diffusion:latest -f Dockerfile .
docker run --gpus all -it -v $(pwd):/home/ubuntu/streamdiffusion stream-diffusion:latest
```

## Quick Start

You can try StreamDiffusion in [`examples`](./examples) directory.

| ![画像3](./assets/demo_02.gif) | ![画像4](./assets/demo_03.gif) |
|:--------------------:|:--------------------:|
| ![画像5](./assets/demo_04.gif) | ![画像6](./assets/demo_05.gif) |

## Real-time Txt2Img Demo

There is an interactive txt2img demo in [`demo/realtime-txt2img`](./demo/realtime-txt2img) directory!

<p align="center">
  <img src="./assets/demo_01.gif" width=100%>
</p>

## minimum example

```python

```

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

# Development Team

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

# Acknowledgements


The video and image demos in this GitHub repository were generated using [kohakuV2](https://civitai.com/models/136268/kohaku-v2) and [SD-Turbo](https://arxiv.org/abs/2311.17042).

Special thanks to Kohaku BlueLeaf ([@KBlueleaf](https://twitter.com/KBlueleaf)) for providing the KohakuV2 model, and to [StabilityAI](https://ja.stability.ai/) for [SD-Turbo](https://arxiv.org/abs/2311.17042).

 KohakuV2 Models can be downloaded from  [Civitai](https://civitai.com/models/136268/kohaku-v2)  and [HuggingFace](https://huggingface.co/stabilityai/sd-turbo).

 [SD-Turbo](https://arxiv.org/abs/2311.17042) is also available on Hugging Face.

# Contributors

<!-- <a href="https://github.com/cumulo-autumn/StreamDiffusion/tree/dev/refactor-examples/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=cumulo-autumn/StreamDiffusion" />
</a> -->
