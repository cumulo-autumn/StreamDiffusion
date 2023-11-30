import os
from typing import *

import ffmpeg
import fire
import PIL.Image
import torch
from diffusers import AutoencoderTiny, LCMScheduler, StableDiffusionPipeline
from tqdm import tqdm

from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.sfast import accelerate_with_stable_fast
from streamdiffusion.image_utils import pil2tensor, postprocess_image


def extract_frames(video_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg.input(video_path).output(f"{output_dir}/%04d.png").run()


def get_frame_rate(video_path: str):
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    return int(video_info["r_frame_rate"].split("/")[0])


def main(input: str, output: str, scale: int = 1):
    if os.path.isdir(output):
        raise ValueError("Output directory already exists")
    frame_rate = get_frame_rate(input)
    extract_frames(input, os.path.join(output, "frames"))
    images = sorted(os.listdir(os.path.join(output, "frames")))

    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file("./model.safetensors").to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()

    sample_image = PIL.Image.open(os.path.join(output, "frames", images[0]))
    width = int(sample_image.width * scale)
    height = int(sample_image.height * scale)

    stream = StreamDiffusion(
        pipe,
        [40, 49],
        torch_dtype=torch.float16,
        width=width,
        height=height,
    )
    stream = accelerate_with_stable_fast(stream)
    stream.prepare(
        "Girl with panda ears wearing a hood",
        num_inference_steps=50,
        generator=torch.manual_seed(2),
    )

    for _ in range(stream.batch_size - 1):
        stream(
            pil2tensor(sample_image.resize((width, height)))
            .detach()
            .clone()
            .to(device=stream.device, dtype=stream.dtype)
        )

    for image_path in tqdm(images + [images[0]] * (stream.batch_size - 1)):
        pil_image = PIL.Image.open(os.path.join(output, "frames", image_path))
        pil_image = pil_image.resize((width, height))
        input_tensor = pil2tensor(pil_image)
        output_x = stream(input_tensor.detach().clone().to(device=stream.device, dtype=stream.dtype))
        output_image = postprocess_image(output_x, output_type="pil")[0]
        output_image.save(os.path.join(output, image_path))

    output_video_path = os.path.join(output, "output.mp4")

    ffmpeg.input(os.path.join(output, "%04d.png"), framerate=frame_rate).output(
        output_video_path, crf=17, pix_fmt="yuv420p", vcodec="libx264"
    ).run()


if __name__ == "__main__":
    fire.Fire(main)
