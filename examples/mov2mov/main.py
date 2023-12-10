import sys
import os
from typing import Literal

import ffmpeg
import fire
import PIL.Image
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
):
    if os.path.isdir(output):
        raise ValueError("Output directory already exists")
    frame_rate = get_frame_rate(input)
    extract_frames(input, os.path.join(output, "frames"))
    images = sorted(os.listdir(os.path.join(output, "frames")))

    sample_image = PIL.Image.open(os.path.join(output, "frames", images[0]))
    width = int(sample_image.width * scale)
    height = int(sample_image.height * scale)

    stream = StreamDiffusionWrapper(
        model_id=model_id,
        t_index_list=[35, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        accerelation=acceleration,
        is_drawing=False,
    )

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    for _ in range(stream.batch_size - 1):
        stream.img2img(sample_image)

    for image_path in tqdm(images + [images[0]] * (stream.batch_size - 1)):
        output_image = stream.img2img(os.path.join(output, "frames", image_path))
        output_image.save(os.path.join(output, image_path))

    output_video_path = os.path.join(output, "output.mp4")

    ffmpeg.input(os.path.join(output, "%04d.png"), framerate=frame_rate).output(
        output_video_path, crf=17, pix_fmt="yuv420p", vcodec="libx264"
    ).run()


if __name__ == "__main__":
    fire.Fire(main)
