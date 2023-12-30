# Txt2Img Example

[English](./README.md) | [日本語](./README-ja.md)

<p align="center">
  <img src="../../assets/demo_01.gif" width=80%>
</p>

This example provides a simple implementation of the use of StreamDiffusion to generate images from text.
You can realtimely generate 4x4 images from text, on changing the input prompt.

## Usage

```bash
pip install -r requirements.txt
cd frontend
pnpm i
pnpm run build
cd ..
python main.py
```

# Acknowledgements

</br>

The video and image demos in this GitHub repository were generated using [kohakuV2](https://civitai.com/models/136268/kohaku-v2) and [SD-Turbo](https://arxiv.org/abs/2311.17042).

Special thanks to Kohaku BlueLeaf ([@KBlueleaf](https://twitter.com/KBlueleaf)) for providing the KohakuV2 model, and to [Stability AI](https://ja.stability.ai/) for [SD-Turbo](https://arxiv.org/abs/2311.17042).

 KohakuV2 Models can be downloaded from  [Civitai](https://civitai.com/models/136268/kohaku-v2)  and [Hugging Face](https://huggingface.co/KBlueLeaf/kohaku-v2.1).

 [SD-Turbo](https://arxiv.org/abs/2311.17042) is also available on [Hugging Face](https://huggingface.co/stabilityai/sd-turbo).
