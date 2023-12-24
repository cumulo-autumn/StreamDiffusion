# StreamDiffusion Examples

[English](./README.md) | [日本語](./README-ja.md)

StreamDiffusion の使用例です。

[README.md](../README.md) に書いてある手順で StreamDiffusion 環境構築及びインストールを行ってください。

TensorRT を用いることで最大パフォーマンスとなります。使用する際は TensorRT を使用した上でコマンドに`--acceleration tensorrt`を付けてください。
`xformers`がデフォルトで使用されていますが、これは最速ではありません。

※※ その他、各種コマンドは最下部の**コマンドオプション**を参考にしてください。

## `screen/`

スクリーンキャプチャをリアルタイムで img2img します。**Windows でのみ動作します。**

スクリプトを実行すると、半透明のウィンドウが出現します。それをキャプチャしたい位置に合わせてエンターキーを押し、キャプチャ範囲を決定してください。

事前に以下のコマンドを実行して依存関係をインストールする必要があります。

```bash
pip install -r screen/requirements.txt
```

### 使用方法

```bash
python screen/main.py
```

## `benchmark/`

StreamDiffusion のパフォーマンス測定を行います。

`benchmark/multi.py`は並列処理を行いますが、`benchmark/single.py`は行いません。

### 使用方法

```bash
python benchmark/multi.py
```

```bash
python benchmark/multi.py --acceleration tensorrt
```

## `optimal-performance/`

TensorRT で最適化された SD-Turbo を用いて txt2img を実行します。

`optimal-performance/multi.py`では RTX4090 に最適化されたバッチ処理を行いますが、`optimal-performance/single.py`は単一バッヂでの処理を行います。

### 使用方法

```bash
python optimal-performance/multi.py
```

```bash
python optimal-performance/single.py
```

## `img2img/`

img2img を実行します。

`img2img/multi.py`は複数の画像が入っているディレクトリを引数として取り、別のディレクトリに img2img の結果を出力します。
`img2img/single.py`は画像一枚の img2img を行います。

### 使用方法

画像 1 枚の img2img:

```bash
python img2img/single.py --input path/to/input.png --output path/to/output.png
```

画像複数枚の img2img(ディレクトリを引数に取ります):

```bash
python img2img/multi.py --input ./input --output-dir ./output
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

vid2vid を実行します。

事前に以下のコマンドを実行して依存関係をインストールする必要があります。

```bash
pip install -r vid2vid/requirements.txt
```

### 使用方法

```bash
python vid2vid/main.py --input path/to/input.mp4 --output path/to/output.mp4
```

# コマンドオプション
### モデル変更
```--model_id_or_path```　引数で使用するモデルを指定できる。
Hugging Face のモデル id を指定することで実行時に Hugging Face からモデルをロードすることができる。<br>
また、ローカルのモデルのパスを指定することでローカルフォルダ内のモデルを使用することも可能である。


例 (Hugging Face) : ```--model_id_or_path "KBlueLeaf/kohaku-v2.1"```<br>
例 (ローカル) : ```--model_id_or_path "C:/stable-diffusion-webui/models/Stable-diffusion/ModelName.safetensor"```

### LoRA 追加
```--lora_dict``` 引数で使用するLoRAを複数指定できる。<br>
```--lora_dict``` は ```"{'LoRA_1 のファイルパス' : LoRA_1 のスケール ,'LoRA_2 のファイルパス' : LoRA_2 のスケール}"``` という形式で指定する。


例 : 
```--lora_dict "{'C:/stable-diffusion-webui/models/Stable-diffusion/LoRA_1.safetensor' : 0.5 ,'E:/ComfyUI/models/LoRA_2.safetensor' : 0.7}"``` 

### Prompt 
```--prompt``` 引数で Prompt を文字列で指定する。

例 : ```--prompt "A cat with a hat"```

### Negative Prompt

```--negative_prompt``` 引数で Negative Prompt を文字列で指定する。<br>
※※ ただし、txt2img ,optimal-performance, vid2vid では使用できない。


例 : ```--negative_prompt "Bad quality"```