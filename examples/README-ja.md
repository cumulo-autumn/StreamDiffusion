# StreamDiffusion Examples

StreamDiffusion の使用例です。

[README.md](../README.md) に書いてある手順で StreamDiffusion 環境構築及びインストールを行ってください。

TensorRT を用いることで最大パフォーマンスとなります。使用する際は実行コマンド `--acceleration tensorrt` を付けてください。
`--acceleration xformers` がデフォルトで使用されていますが、これは最速ではありません。


## `benchmark/`

StreamDiffusion のパフォーマンス測定を行います。

`benchmark/multi.py` は並列処理を行いますが、`benchmark/single.py`は行いません。

### 使用方法

```bash
python benchmark/multi.py
```

```bash
python benchmark/multi.py --acceleration tensorrt
```

## `img2img/`

Image-to-imageを実行します。

`img2img/multi.py` は複数の画像が入っているディレクトリを引数として取り、別のディレクトリに Image-to-image  の結果を出力します。
`img2img/single.py` は画像一枚の Img2img です。

### 使用方法

画像一枚を用いた Image-to-image :

```bash
python img2img/single.py --input path/to/input.png --output path/to/output.png
```

複数枚画像の Image-to-image :

```bash
python img2img/multi.py --input ./input --output-dir ./output
```

## `optimal-performance/`

TensorRT で最適化された SD-Turbo を用いて text-to-image を実行します。

`optimal-performance/multi.py` では RTX4090 に最適化されたバッチ処理を行いますが、`optimal-performance/single.py`は単一バッヂでの処理を行います。

### 使用方法

```bash
python optimal-performance/multi.py
```

```bash
python optimal-performance/single.py
```

## `screen/`

**This script only works on Windows.**

スクリーンキャプチャを用いたリアルタイムの image-to-image です。**Windowsでのみ動作します。**

動作のために、以下のコマンドを用いて依存関係をインストールする必要があります。

```bash
pip install -r screen/requirements.txt
```

### 使用方法

```bash
python screen/main.py
```

## `txt2img/`

text-to-image

`txt2img/multi.py` は Prompt から複数の画像を生成し、`txt2img/single.py` は一枚の画像を生成します。

### 使用方法

一枚だけ生成する場合:

```bash
python txt2img/single.py --output output.png --prompt "A cat with a hat"
```

複数の画像を生成する場合:

```bash
python txt2img/multi.py --output ./output --prompt "A cat with a hat"
```

## `vid2vid/`

video-to-videoを実行します。

動作のために、以下のコマンドを用いて依存関係をインストールする必要があります。

```bash
pip install -r vid2vid/requirements.txt
```

### 使用方法

```bash
python vid2vid/main.py --input path/to/input.mp4 --output path/to/output.mp4
```

