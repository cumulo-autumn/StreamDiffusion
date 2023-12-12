import os
import sys
from typing import Literal

import ffmpeg
import fire
import torch
from torchvision.io import read_video, write_video
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from wrapper import StreamDiffusionWrapper


def extract_frames(video_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg.input(video_path).output(f"{output_dir}/%04d.png").run()


def get_frame_rate(video_path: str):
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    return int(video_info["r_frame_rate"].split("/")[0])


def main(
    input: str,
    output: str = "output",
    model_id: str = "KBlueLeaf/kohaku-v2.1",
    prompt: str = "Girl with panda ears wearing a hood",
    scale: float = 1.0,
    acceleration: Literal["none", "xformers", "sfast", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
):
    video_info = read_video(input)
    video = video_info[0] / 255
    fps = video_info[2]["video_fps"]
    width = int(video.shape[1] * scale)
    height = int(video.shape[2] * scale)

    stream = StreamDiffusionWrapper(
        model_id=model_id,
        t_index_list=[35, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        is_drawing=False,
        mode="img2img",
        output_type="pt",
        enable_similar_image_filter=True,
        similar_image_filter_threshold=0.90,
        use_denoising_batch = use_denoising_batch,
    )

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    video_result = torch.zeros(video.shape[0], width, height, 3)

    for _ in range(stream.batch_size - 1):
        stream(image=video[0].permute(2, 0, 1))

    for i in tqdm(range(video.shape[0])):
        output_image = stream(video[i].permute(2, 0, 1))
        video_result[i] = output_image.permute(1, 2, 0)

    video_result = video_result * 255
    write_video(f"{output}.mp4", video_result, fps=fps)


if __name__ == "__main__":
    fire.Fire(main)