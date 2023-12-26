# Img2Img Example

[English](./README.md) 

<p align="center">
  <img src="../../assets/img2img1.gif" width=80%>
</p>

<p align="center">
  <img src="../../assets/img2img2.gif" width=80%>
</p>


This example, based on this [MPJEG server](https://github.com/radames/Real-Time-Latent-Consistency-Model/), runs image-to-image with a live webcam feed or screen capture on a web browser.

## Usage
You need Node.js 18+ and Python 3.10 to run this example.

```bash
cd frontend
npm i
npm run build
pip install -r requirements.txt
cd ..
python main.py  --acceleration tensorrt   
```
