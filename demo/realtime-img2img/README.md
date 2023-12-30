# Img2Img Example

[English](./README.md) | [日本語](./README-ja.md)

<p align="center">
  <img src="../../assets/img2img1.gif" width=80%>
</p>

<p align="center">
  <img src="../../assets/img2img2.gif" width=80%>
</p>


This example, based on this [MPJEG server](https://github.com/radames/Real-Time-Latent-Consistency-Model/), runs image-to-image with a live webcam feed or screen capture on a web browser.

## Usage
You need Node.js 18+ and Python 3.10 to run this example.
Please make sure you've installed all dependencies according to the [installation instructions](../../README.md#installation).

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

then open `http://0.0.0.0:7860` in your browser.

### Running with Docker

```bash
docker build -t img2img .
docker run -ti -e ENGINE_DIR=/data -e HF_HOME=/data -v ~/.cache/huggingface:/data  -p 7860:7860 --gpus all img2img
```

Where `ENGINE_DIR` and `HF_HOME` set a local cache directory, making it faster to restart the docker container.