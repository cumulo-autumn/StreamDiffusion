# StreamDiffusion Examples

[English](./README.md) | [日本語](./README-ja.md)

StreamDiffusionの使用例です。

[README.md](../README.md) に書いてある手順でStreamDiffusion環境構築及びインストールを行ってください。

TensorRTを用いることで最大パフォーマンスとなります。使用する際はTensorRTを使用した上でコマンドに`--acceleration tensorrt`を付けてください。
`xformers`がデフォルトで使用されていますが、これは最速ではありません。


## `benchmark/`

StreamDiffusionのパフォーマンス測定を行います。

`benchmark/multi.py`は並列処理を行いますが、`benchmark/single.py`は行いません。

### 使用方法

```bash
python benchmark/multi.py
```

```bash
python benchmark/multi.py --acceleration tensorrt
```

## `img2img/`

img2imgを実行します。

`img2img/multi.py`は複数の画像が入っているディレクトリを引数として取り、別のディレクトリにimg2imgの結果を出力します。
`img2img/single.py`は画像一枚のimg2imgを行います。

### 使用方法

画像1枚のimg2img:

```bash
python img2img/single.py --input path/to/input.png --output path/to/output.png
```

画像複数枚のimg2img(ディレクトリを引数に取ります):

```bash
python img2img/multi.py --input ./input --output-dir ./output
```

## `optimal-performance/`

TensorRTで最適化されたSD-Turboを用いてtxt2imgを実行します。

`optimal-performance/multi.py`ではRTX4090に最適化されたバッチ処理を行いますが、`optimal-performance/single.py`は単一バッヂでの処理を行います。

### 使用方法

```bash
python optimal-performance/multi.py
```

```bash
python optimal-performance/single.py
```

## `screen/`

スクリーンキャプチャをリアルタイムでimg2imgします。**Windowsでのみ動作します。**

事前に以下のコマンドを実行して依存関係をインストールする必要があります。

```bash
pip install -r screen/requirements.txt
```

### 使用方法

```bash
python screen/main.py
```

## `txt2img/`

`txt2img/multi.py`はプロンプトから複数の画像を生成し、`txt2img/single.py`は一枚の画像を生成します。

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

vid2vidを実行します。

事前に以下のコマンドを実行して依存関係をインストールする必要があります。

```bash
pip install -r vid2vid/requirements.txt
```

### 使用方法

```bash
python vid2vid/main.py --input path/to/input.mp4 --output path/to/output.mp4
```

