# StreamDiffusion Examples

[English](./README.md) | [日本語](./README-ja.md)

Examples of StreamDiffusion.

If you want to maximize performance, you need to install with following steps explained in [README.md](../README.md) in the root directory, and use `--acceleration tensorrt` option at the end of each command. At default, StreamDiffusion uses `xformers` for acceleration, which is not the fastest option.

## `screen/`

Take a screen capture and process it. **This script only works on Windows.**

You need to install extra dependencies for this script as follows:

```bash
pip install -r screen/requirements.txt
```

### Usage

```bash
python screen/main.py
```

## `benchmark/`

Just measure the performance of StreamDiffusion.

`benchmark/multi.py` spawns multiple processes for postprocessing and `benchmark/single.py` does not.

### Usage

```bash
python benchmark/multi.py
```

With TensorRT acceleration:

```bash
python benchmark/multi.py --acceleration tensorrt
```

## `optimal-performance/`

Using SD-Turbo and TensorRT, perform text-to-image with optimal performance.

`optimal-performance/multi.py` is optimized for RTX4090 and performs batch processing, while `optimal-performance/single.py` does not.

### Usage

```bash
python optimal-performance/multi.py
```

```bash
python optimal-performance/single.py
```

## `img2img/`

Perform image-to-image.

`img2img/multi.py` takes a directory of input images and outputs to another directory as arguments, and `img2img/single.py` takes a single image.

### Usage

Image-to-image for a single image:

```bash
python img2img/single.py --input path/to/input.png --output path/to/output.png
```

Image-to-image for multiple images:

```bash
python img2img/multi.py --input ./input --output-dir ./output
```

## `txt2img/`

Perform text-to-image.

`txt2img/multi.py` generates multiple images from a single prompt, and `txt2img/single.py` generates a single image.

### Usage

Text-to-image for a single image:

```bash
python txt2img/single.py --output output.png --prompt "A cat with a hat"
```

Text-to-image for multiple images:

```bash
python txt2img/multi.py --output ./output --prompt "A cat with a hat"
```

## `vid2vid/`

Perform video-to-video conversion.

You need to install extra dependencies for this script as follows:

```bash
pip install -r vid2vid/requirements.txt
```

### Usage

```bash
python vid2vid/main.py --input path/to/input.mp4 --output path/to/output.mp4
```

