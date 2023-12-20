# Txt2Img Example

<p align="center">
  <img src="./assets/example.gif" width=80%>
</p>

This example provides a simple implementation of the use of StreamDiffusion to generate images from text.
You can realtimely generate 4x4 images from text, on changing the input prompt.

## Usage

```bash
pip install -r requirements.txt
cd view
npm i
npm run build
cd ../server
python main.py
```

## Docker

Build
`GITHUB_TOKEN` is temp until project is public
```bash
docker build --secret id=GITHUB_TOKEN,src=./github_token.txt -t realtime-txt2img .
```

Run
```bash
docker run -ti -p 9090:9090 -e HF_HOME=/data -v ~/.cache/huggingface:/data  --gpus all realtime-txt2img
```

`-e HF_HOME=/data -v ~/.cache/huggingface:/data` is used to mount your local huggingface cache to the container, so that you don't need to download the model again.
