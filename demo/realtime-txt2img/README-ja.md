# Txt2Img Example

[English](./README.md) | [日本語](./README-ja.md)

<p align="center">
  <img src="../../assets/demo_01.gif" width=80%>
</p>

StreamDiffusion を用いた GUI を提供します。
入力プロンプトを変更すると、テキストから 4x4 の画像をリアルタイムに生成することができます。

## 使用方法

以下のコマンドを順番に実行してください。

```bash
pip install -r requirements.txt
cd frontend
pnpm i
pnpm run build
cd ..
python main.py
```

# 謝辞

この GitHubリポジトリ にある動画と画像のデモは、[kohakuV2](https://civitai.com/models/136268/kohaku-v2)と[SD-Turbo](https://arxiv.org/abs/2311.17042)を使用して生成されました。

KohakuV2 モデルを提供していただいたKohaku BlueLeaf 様 ([@KBlueleaf](https://twitter.com/KBlueleaf))、[SD-Turbo](https://arxiv.org/abs/2311.17042)を提供していただいた[Stability AI](https://ja.stability.ai/)様に心より感謝いたします。

KohakuV2 モデルは [Civitai](https://civitai.com/models/136268/kohaku-v2) と [Hugging Face](https://huggingface.co/KBlueLeaf/kohaku-v2.1) からダウンロードでき、[SD-Turbo](https://arxiv.org/abs/2311.17042) は [Hugging Face](https://huggingface.co/stabilityai/sd-turbo) で使用可能です。
