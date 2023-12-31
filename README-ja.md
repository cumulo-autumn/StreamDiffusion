# StreamDiffusion

[English](./README.md) | [日本語](./README-ja.md)

<p align="center">
  <img src="./assets/demo_07.gif" width=90%>
  <img src="./assets/demo_09.gif" width=90%>
</p>

# StreamDiffusion: A Pipeline-Level Solution for Real-Time Interactive Generation

**Authors:** [Akio Kodaira](https://www.linkedin.com/in/akio-kodaira-1a7b98252/), [Chenfeng Xu](https://www.chenfengx.com/), Toshiki Hazama, [Takanori Yoshimoto](https://twitter.com/__ramu0e__), [Kohei Ohno](https://www.linkedin.com/in/kohei--ohno/), [Shogo Mitsuhori](https://me.ddpn.world/), [Soichi Sugano](https://twitter.com/toni_nimono), [Hanying Cho](https://twitter.com/hanyingcl), [Zhijian Liu](https://zhijianliu.com/), [Kurt Keutzer](https://scholar.google.com/citations?hl=en&user=ID9QePIAAAAJ)


StreamDiffusionは、リアルタイム画像生成を実現するために最適化されたパイプラインです。従来の画像生成パイプラインと比べて飛躍的な速度向上を実現しました。

[![arXiv](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)](https://arxiv.org/abs/2312.12491)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/papers/2312.12491)

StreamDiffusionの開発にあたり、丁寧なサポート、有意義なフィードバックと議論をしていただいた [Taku Fujimoto](https://twitter.com/AttaQjp) 様と [Radamés Ajna](https://twitter.com/radamar) 様、そして Hugging Face チームに心より感謝いたします。

## 主な特徴

1. **Stream Batch**
   - デノイジングバッチ処理によるデータ処理の効率化

2. **Residual Classifier-Free Guidance** - [詳細](#residual-cfg-rcfg)
   - 計算の冗長性を最小限に抑えるCFG

3. **Stochastic Similarity Filter** - [詳細](#stochastic-similarity-filter)
   - 類似度によるフィルタリングでGPUの使用効率を最大化

4. **IO Queues**
   - 入出力操作を効率的に管理し、よりスムーズな実行を実現

5. **Pre-Computation for KV-Caches**
   - 高速処理のためのキャッシュ戦略を最適化します。

6. **Model Acceleration Tools**
   - モデルの最適化とパフォーマンス向上のための様々なツールの利用

**GPU: RTX 4090**, **CPU: Core i9-13900K**, **OS: Ubuntu 22.04.3 LTS**　環境で StreamDiffusion pipeline を用いて 画像を生成した場合、以下のような結果が得られました。

|model                | Denoising Step      |  fps on Txt2Img      |  fps on Img2Img      |
|:-------------------:|:-------------------:|:--------------------:|:--------------------:|
|SD-turbo             | 1              | 106.16                    | 93.897               |
|LCM-LoRA <br>+<br> KohakuV2| 4        | 38.023                    | 37.133               |

_Feel free to explore each feature by following the provided links to learn more about StreamDiffusion's capabilities. If you find it helpful, please consider citing our work:_

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

### Step0: リポジトリのクローン

```bash
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
```

### Step1: 仮想環境構築

anaconda、pip、または後述するDockerで仮想環境を作成します。

anacondaを用いる場合

```bash
conda create -n streamdiffusion python=3.10
conda activate streamdiffusion
```

pipを用いる場合

```cmd
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux
source .venv/bin/activate
```

### Step2: PyTorchのインストール

使用するGPUのCUDAバージョンに合わせてPyTorchをインストールしてください。

CUDA 11.8

```bash
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu118
```

CUDA 12.1

```bash
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
```

詳しくは[こちら](https://pytorch.org/)


### Step3: StreamDiffusionのインストール

StreamDiffusionをインストール

#### ユーザー向け

```bash
#最新バージョン (推奨)
pip install git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt]


#もしくは


#リリースバージョン
pip install streamdiffusion[tensorrt]
```

TensorRT拡張をインストール


```bash
python -m streamdiffusion.tools.install-tensorrt
```
(Only for Windows)リリースバージョン(`pip install streamdiffusion[tensorrt]`)ではpywin32のインストールが別途必要です。
```bash
pip install --force-reinstall pywin32
```

#### 開発者向け

```bash
python setup.py develop easy_install streamdiffusion[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
```

### Dockerの場合(TensorRT対応)

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

## Real-Time Txt2Img Demo

リアルタイムのtxt2imgデモは [`demo/realtime-txt2img`](./demo/realtime-txt2img)にあります。

<p align="center">
  <img src="./assets/demo_01.gif" width=80%>
</p>

## Real-Time Img2Img Demo

Webカメラを使ったリアルタイムのimg2imgデモは [`demo/realtime-img2img`](./demo/realtime-img2img)にあります。

<p align="center">
  <img src="./assets/img2img1.gif" width=100%>
</p>

## 使用例
シンプルなStreamDiffusionの使用例を取り上げる. より詳細かつ様々な使用例は[`examples`](./examples)を参照してください。



### Image-to-Image
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

# Diffusers pipelineをStreamDiffusionにラップ
stream = StreamDiffusion(
    pipe,
    t_index_list=[32, 45],
    torch_dtype=torch.float16,
)

# 読み込んだモデルがLCMでなければマージする
stream.load_lcm_lora()
stream.fuse_lora()
# Tiny VAEで高速化
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
# xformersで高速化
pipe.enable_xformers_memory_efficient_attention()


prompt = "1girl with dog hair, thick frame glasses"
# streamを準備する
stream.prepare(prompt)

# 画像を読み込む
init_image = load_image("assets/img2img_example.png").resize((512, 512))

# Warmup >= len(t_index_list) x frame_buffer_size
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

### Text-to-Image
```python
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1").to(
    device=torch.device("cuda"),
    dtype=torch.float16,
)

# Diffusers pipelineをStreamDiffusionにラップ
# text2imageにおいてはより長いステップ(len(t_index_list))を要求する
# text2imageにおいてはcfg_type="none"が推奨される
stream = StreamDiffusion(
    pipe,
    t_index_list=[0, 16, 32, 45],
    torch_dtype=torch.float16,
    cfg_type="none",
)

# 読み込んだモデルがLCMでなければマージする
stream.load_lcm_lora()
stream.fuse_lora()
# Tiny VAEで高速化
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
# xformersで高速化
pipe.enable_xformers_memory_efficient_attention()


prompt = "1girl with dog hair, thick frame glasses"
# streamを準備する
stream.prepare(prompt)

# Warmup >= len(t_index_list) x frame_buffer_size
for _ in range(4):
    stream()

# 実行
while True:
    x_output = stream.txt2img()
    postprocess_image(x_output, output_type="pil")[0].show()
    input_response = input("Press Enter to continue or type 'stop' to exit: ")
    if input_response == "stop":
        break
```
SD-Turboを使用するとさらに高速化も可能である

### More fast generation
上のコードの以下の部分を書き換えることで、より高速な生成が可能である。
```python
pipe.enable_xformers_memory_efficient_attention()
```
以下に書き換える
```python
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

stream = accelerate_with_tensorrt(
    stream, "engines", max_batch_size=2,
)
```
ただし、TensorRTのインストールとエンジンのビルドに時間を要する。

## オプション

### Stochastic Similarity Filter

![demo](assets/demo_06.gif)

Stochastic Similarity Filterは動画入力時、前フレームからあまり変化しないときの変換処理を減らすことで、上のGIFの赤枠の様にGPUの負荷を軽減する。使用方法は以下のとおりである。

```python
stream = StreamDiffusion(
    pipe,
    [32, 45],
    torch_dtype=torch.float16,
)
stream.enable_similar_image_filter(
    similar_image_filter_threshold,
    similar_image_filter_max_skip_frame,
)
```

関数で設定できる引数として以下がある。

#### `similar_image_filter_threshold`

- 処理を休止する前フレームと現フレームの類似度の閾値

#### `similar_image_filter_max_skip_frame`

- 休止中に変換を再開する最大の間隔

### Residual CFG (RCFG)

![rcfg](assets/cfg_conparision.png)

RCFGはCFG使用しない場合と比較し、競争力のある計算量で近似的にCFGを実現させる方法である。StreamDiffusionの引数cfg_typeから指定可能である。

RCFGは二種類あり、negative promptの指定項目なしのRCFG Self-Negativeとnegative promptが指定可能なOnetime-Negativeが利用可能である。計算量はCFGなしの計算量をN、通常のCFGありの計算量を２Nとしたとき、RCFG Self-NegativeはN回で、Onetime-NegativeはN+1回で計算できる。

The usage is as follows:

```python
# CFG なし
cfg_type = "none"

# 通常のCFG
cfg_type = "full"

# RCFG Self-Negative
cfg_type = "self"

# RCFG Onetime-Negative
cfg_type = "initialize"

stream = StreamDiffusion(
    pipe,
    [32, 45],
    torch_dtype=torch.float16,
    cfg_type=cfg_type,
)

stream.prepare(
    prompt="1girl, purple hair",
    guidance_scale=guidance_scale,
    delta=delta,
)
```

deltaはRCFGの効きをマイルドにする効果を持つ

## 開発チーム

[Aki](https://twitter.com/cumulo_autumn),
[Ararat](https://twitter.com/AttaQjp),
[Chenfeng Xu](https://twitter.com/Chenfeng_X),
[ddPn08](https://twitter.com/ddPn08),
[kizamimi](https://twitter.com/ArtengMimi),
[ramune](https://twitter.com/__ramu0e__),
[teftef](https://twitter.com/hanyingcl),
[Tonimono](https://twitter.com/toni_nimono),
[Verb](https://twitter.com/IMG_5955),

(*alphabetical order)
</br>

## 謝辞

この GitHubリポジトリ にある動画と画像のデモは、[LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) + [KohakuV2](https://civitai.com/models/136268/kohaku-v2)と[SD-Turbo](https://arxiv.org/abs/2311.17042)を使用して生成されました。

LCM-LoRAを提供していただいた[LCM-LoRA authors](https://latent-consistency-models.github.io/)、KohakuV2 モデルを提供していただいたKohaku BlueLeaf 様 ([@KBlueleaf](https://twitter.com/KBlueleaf))、[SD-Turbo](https://arxiv.org/abs/2311.17042)を提供していただいた[Stability AI](https://ja.stability.ai/)様に心より感謝いたします。

KohakuV2 モデルは [Civitai](https://civitai.com/models/136268/kohaku-v2) と [Hugging Face](https://huggingface.co/KBlueLeaf/kohaku-v2.1) からダウンロードでき、[SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) は Hugging Faceで使用可能です。


## Contributors

<a href="https://github.com/cumulo-autumn/StreamDiffusion/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=cumulo-autumn/StreamDiffusion" />
</a>
