# StreamDiffusion

[English](./README.md) | [日本語](./README-ja.md)

<p align="center">
  <img src="./assets/demo_07.gif" width=90%>
  <img src="./assets/demo_09.gif" width=90%>
</p>

# StreamDiffusion: A Pipeline-Level Solution for Real-Time Interactive Generation

**Authors:** [Akio Kodaira](https://www.linkedin.com/feed/), [Chenfeng Xu](https://www.chenfengx.com/), Toshiki Hazama, Takanori Yoshimoto, [Kohei Ohno](https://www.linkedin.com/in/kohei--ohno/), [Shogo Mitsuhori](https://me.ddpn.world/), Soichi Sugano, Hanying Cho, [Zhijian Liu](https://zhijianliu.com/), [Kurt Keutzer](https://scholar.google.com/citations?hl=en&user=ID9QePIAAAAJ)


StreamDiffusionは、リアルタイム画像生成を実現するために最適化された、革新的な画像生成パイプラインです。StreamDiffusionは、従来の画像生成パイプラインと比べて飛躍的な速度向上を実現しました。

[![arXiv](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)](https://arxiv.org/abs/2312.12491)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/papers/2312.12491)

StreamDiffusionの開発にあたり、丁寧なサポート、有意義なフィードバックと議論をしていただいた [Taku Fujimoto](https://twitter.com/AttaQjp) 様と [Radamés Ajna](https://twitter.com/radamar) 様、そして Huggingface チームに心より感謝いたします。

## 主な特徴

1. **Stream Batch** - [詳細](#stream-batching-link)
   - バッチ処理によるデータ処理の効率化を行っています。

2. **Residual Classifier-Free Guidance** - [詳細](#residual-classifier-free-guidance-link)
   - 計算の冗長性を最小限に抑える改良されたガイダンスメカニズム。

3. **Stochastic Similarity Filter** - [詳細](#stochastic-similarity-filtering-link)
   - 高度なフィルタリング技術によりGPUの利用効率を向上させます。

4. **IO Queues** - [詳細](#io-queues-link)
   - 入出力操作を効率的に管理し、よりスムーズな実行を実現します。

5. **Pre-computation for KV-Caches** - [詳細](#pre-computation-for-kv-caches-link)
   - 高速処理のためのキャッシュ戦略を最適化します。

6. **Model Acceleration Tools**
   - モデルの最適化とパフォーマンス向上のための様々なツールを利用できます。

StreamDiffusionの機能をより詳しく知るために、提供されているリンクをたどって各機能を自由に探索してください。お役に立ちましたら、ぜひ引用をご検討ください：

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

## インストール

### 環境構築

anaconda または pip で仮想環境を作成してください。

anaconda を用いる場合

```bash
conda create -n streamdiffusion python=3.10
conda activate streamdiffusion
```

pip を用いる場合

```cmd
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux
source .venv/bin/activate
```

### Step1: PyTorch のインストール

使用する GPU の CUDA バージョンに合わせて PyTorch をインストールしてください。

CUDA 11.8

```bash
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu118
```

CUDA 12.1

```bash
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
```

詳しくは[こちら](https://pytorch.org/)


### Step2: StreamDiffusion のインストール

#### 非開発者向け

StreamDiffusion をインストール

```bash
pip install git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=streamdiffusion
```

tensorrt をインストール

```bash
python -m streamdiffusion.tools.install-tensorrt
```

#### 開発者向け

```bash
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
python setup.py develop easy_install streamdiffusion[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
```

## Docker　の場合 (TensorRT 対応)

```bash
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
cd StreamDiffusion
docker build -t stream-diffusion:latest -f Dockerfile .
docker run --gpus all -it -v $(pwd):/home/ubuntu/streamdiffusion stream-diffusion:latest
```

## 動作例

 [`examples`](./examples) からサンプルを実行できます。

| ![画像3](./assets/demo_02.gif) | ![画像4](./assets/demo_03.gif) |
|:--------------------:|:--------------------:|
| ![画像5](./assets/demo_04.gif) | ![画像6](./assets/demo_05.gif) |

具体的な詳細設定及びユーザカスタマイズは以下をお読みください。

## リアルタイム Txt2Img デモ

リアルタイムの txt2img デモは [`demo/realtime-txt2img`](./demo/realtime-txt2img) ディレクトリにあります。

<p align="center">
  <img src="./assets/demo_01.gif" width=80%>
</p>

## minimum example
```python
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1").to(
    device=torch.device("cuda"),
    dtype=torch.float16,
)

# StreamDiffusionでパイプラインをラップ
stream = StreamDiffusion(
    pipe,
    t_index_list=[32, 45],
    do_add_noise=True,
    torch_dtype=torch.float16,
)

# 読みこんだモデルがLCMではない場合、LCMをマージする
stream.load_lcm_lora()
stream.fuse_lora()

# Tiny VAEを使用しさらに高速化
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

# xformersを使用してさらに高速化
# TensorRTが利用可能であればそちらを推奨
pipe.enable_xformers_memory_efficient_attention()

# プロンプトの設定
prompt = "1girl with dog hair, thick frame glasses"
stream.prepare(prompt)

# 画像の読み込み
init_image = load_image("assets\img2img_example.png").resize((512, 512))

# ワームアップ: >= len(t_index_list) x frame_buffer_size
for _ in range(2):
    stream(init_image)

# 実行
while True:
    x_output = stream(init_image)
    postprocess_image(x_output, output_type="pil")[0].show()
    input_response = input("Press Enter to continue or type 'stop' to exit: ")
    if input_response == "stop":
        break
```

## ユーザカスタマイズ

下記は、ローカル環境にて StreamDiffusion を実行する際のサンプルコードです。

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
from streamdiffusion.image_utils import pil2tensor, postprocess_image


def download_image(url: str):
    response = requests.get(url)
    image = PIL.Image.open(io.BytesIO(response.content))
    return image


def run(
    wamup: int = 10,
    iterations: int = 50,
    prompt: str = "1girl with brown dog ears, thick frame glasses",
    lcm_lora: bool = True,
    tiny_vae: bool = True,
    acceleration: Optional[Literal["xformers", "sfast", "tensorrt"]] = None,
):
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file("./model.safetensors").to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    stream = StreamDiffusion(
        pipe,
        [32, 45],
        torch_dtype=torch.float16,
    )

    if lcm_lora:
        stream.load_lcm_lora()
        stream.fuse_lora()

    if tiny_vae:
        stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

    if acceleration == "xformers":
        pipe.enable_xformers_memory_efficient_attention()
    elif acceleration == "tensorrt":
        from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

        stream = accelerate_with_tensorrt(
            stream, "engines", max_batch_size=2, engine_build_options={"build_static_batch": True}
        )
    elif acceleration == "sfast":
        from streamdiffusion.acceleration.sfast import accelerate_with_stable_fast

        stream = accelerate_with_stable_fast(stream)

    stream.prepare(
        prompt,
        num_inference_steps=50,
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
```

# 開発チーム

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

# 謝辞

この GitHubリポジトリ にある動画と画像のデモは、[kohakuV2](https://civitai.com/models/136268/kohaku-v2)と[SD-Turbo](https://arxiv.org/abs/2311.17042)を使用して生成されました。

KohakuV2 モデルを提供していただいたKohaku BlueLeaf 様 ([@KBlueleaf](https://twitter.com/KBlueleaf))、[SD-Turbo](https://arxiv.org/abs/2311.17042)を提供していただいた[StabilityAI](https://ja.stability.ai/)様に心より感謝いたします。

KohakuV2 モデルは [Civitai](https://civitai.com/models/136268/kohaku-v2) と [HuggingFace](https://huggingface.co/stabilityai/sd-turbo) からダウンロードでき、[SD-Turbo](https://arxiv.org/abs/2311.17042) は Hugging Faceで使用可能です。。


# Contributors

<!-- <a href="https://github.com/cumulo-autumn/StreamDiffusion/tree/dev/refactor-examples/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=cumulo-autumn/StreamDiffusion" />
</a> -->
