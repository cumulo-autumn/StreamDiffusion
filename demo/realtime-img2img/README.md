---
title: Real-Time Latent Consistency Model Image-to-Image ControlNet
emoji: 🖼️🖼️
colorFrom: gray
colorTo: indigo
sdk: docker
pinned: false
suggested_hardware: a10g-small
disable_embedding: true
---

# Real-Time Latent Consistency Model

This demo showcases [Latent Consistency Model (LCM)](https://latent-consistency-models.github.io/) using [Diffusers](https://huggingface.co/docs/diffusers/using-diffusers/lcm) with a MJPEG stream server. You can read more about LCM + LoRAs with diffusers [here](https://huggingface.co/blog/lcm_lora).

You need a webcam to run this demo. 🤗

See a collecting with live demos [here](https://huggingface.co/collections/latent-consistency/latent-consistency-model-demos-654e90c52adb0688a0acbe6f)

## Running Locally

You need CUDA and Python 3.10, Node > 19, Mac with an M1/M2/M3 chip or Intel Arc GPU


## Install

```bash
python -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
# fastest pipeline
python run.py --reload --pipeline img2imgSD21Turbo 
 ```

# Pipelines
You can build your own pipeline following examples here [here](pipelines),
don't forget to fuild the frontend first 
```bash
cd frontend && npm install && npm run build && cd ..
```

# LCM
### Image to Image

```bash
python run.py --reload --pipeline img2img 
```

# LCM
### Text to Image

```bash
python run.py --reload --pipeline txt2img 
```

### Image to Image ControlNet Canny


```bash
python run.py --reload --pipeline controlnet 
```


# LCM + LoRa

Using LCM-LoRA, giving it the super power of doing inference in as little as 4 steps. [Learn more here](https://huggingface.co/blog/lcm_lora) or [technical report](https://huggingface.co/papers/2311.05556)



### Image to Image ControlNet Canny LoRa

```bash
python run.py --reload --pipeline controlnetLoraSD15
```
or SDXL, note that SDXL is slower than SD15 since the inference runs on 1024x1024 images

```bash
python run.py --reload --pipeline controlnetLoraSDXL
```

### Text to Image

```bash
python run.py --reload --pipeline txt2imgLora
```

or 

```bash
python run.py --reload --pipeline txt2imgLoraSDXL
```


### Setting environment variables


`TIMEOUT`: limit user session timeout  
`SAFETY_CHECKER`: disabled if you want NSFW filter off  
`MAX_QUEUE_SIZE`: limit number of users on current app instance  
`TORCH_COMPILE`: enable if you want to use torch compile for faster inference works well on A100 GPUs
`USE_TAESD`: enable if you want to use Autoencoder Tiny

If you run using `bash build-run.sh` you can set `PIPELINE` variables to choose the pipeline you want to run

```bash
PIPELINE=txt2imgLoraSDXL bash build-run.sh
```

and setting environment variables

```bash
TIMEOUT=120 SAFETY_CHECKER=True MAX_QUEUE_SIZE=4 python run.py --reload --pipeline txt2imgLoraSDXL
```

If you're running locally and want to test it on Mobile Safari, the webserver needs to be served over HTTPS, or follow this instruction on my [comment](https://github.com/radames/Real-Time-Latent-Consistency-Model/issues/17#issuecomment-1811957196)

```bash
openssl req -newkey rsa:4096 -nodes -keyout key.pem -x509 -days 365 -out certificate.pem
python run.py --reload --ssl-certfile=certificate.pem --ssl-keyfile=key.pem
```

## Docker

You need NVIDIA Container Toolkit for Docker, defaults to `controlnet``

```bash
docker build -t lcm-live .
docker run -ti -p 7860:7860 --gpus all lcm-live
```

reuse models data from host to avoid downloading them again, you can change `~/.cache/huggingface` to any other directory, but if you use hugingface-cli locally, you can share the same cache

```bash
docker run -ti -p 7860:7860 -e HF_HOME=/data -v ~/.cache/huggingface:/data  --gpus all lcm-live
```
 

or with environment variables

```bash
docker run -ti -e PIPELINE=txt2imgLoraSDXL -p 7860:7860 --gpus all lcm-live
```
# Development Mode


```bash
python run.py --reload  
```

# Demo on Hugging Face

https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model

https://github.com/radames/Real-Time-Latent-Consistency-Model/assets/102277/c4003ac5-e7ff-44c0-97d3-464bb659de70
