# Txt2Img Example

<p align="center">
  <img src="../../assets/demo_01.gif" width=80%>
</p>

StreamDiffusion を用いた GUI を提供します。
入力プロンプトを変更すると、テキストから 4x4 の画像をリアルタイムに生成することができます。

## 使用方法

以下のコマンドを順番に実行してください。

```bash
pip install -r requirements.txt
cd view
npm i
npm start &
cd ../server
python main.py
```

## Docker

Build

`GITHUB_TOKEN` を各自変更してください。
```bash
docker build --secret id=GITHUB_TOKEN,src=./github_token.txt -t realtime-txt2img .
```

実行
```bash
docker run -ti -p 9090:9090 -e HF_HOME=/data -v ~/.cache/huggingface:/data  --gpus all realtime-txt2img
```


`-e HF_HOME=/data -v ~/.cache/huggingface:/data` はローカルの huggingface キャッシュをコンテナにマウントするために使用されます。
