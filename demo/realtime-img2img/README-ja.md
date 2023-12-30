# Img2Img Example

[English](./README.md) | [日本語](./README-ja.md)

<p align="center">
  <img src="../../assets/img2img1.gif" width=80%>
</p>

<p align="center">
  <img src="../../assets/img2img2.gif" width=80%>
</p>


こちらの [MPJEG server](https://github.com/radames/Real-Time-Latent-Consistency-Model/)のデモを元にした、webカメラやスクリーンキャプチャを用いたimage-to-imageのデモです。Promptを変更することで、リアルタイムにプロンプトの効果を生成画像に反映することが出来ます。

## Usage
こちらのデモを実行するには Node.js 18+が必要です。また、Python 3.10以外での動作は未確認です。
[installation instructions](../../README.md#installation)に従って、事前に必要なライブラリをインストールしてください。

```bash
cd frontend
npm i
npm run build
cd ..
pip install -r requirements.txt
python main.py  --acceleration tensorrt
```

or

```
chmod +x start.sh
./start.sh
```

上記のコマンドを実行した後 `http://0.0.0.0:7860` をブラウザで開いてください。
(※ `http://0.0.0.0:7860`で上手く動作しない場合は、`http://localhost:7860`を試してみてください)

### Running with Docker

```bash
docker build -t img2img .
docker run -ti -e ENGINE_DIR=/data -e HF_HOME=/data -v ~/.cache/huggingface:/data  -p 7860:7860 --gpus all img2img
```

ここで、`ENGINE_DIR`と`HF_HOME`はローカルのキャッシュディレクトリを設定し、dockerコンテナの再起動を高速化します。