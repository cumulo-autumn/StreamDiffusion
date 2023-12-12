import os
import sys
from typing import Literal

import ffmpeg
import fire
import PIL.Image
import torch
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
        acceleration=acceleration,
        is_drawing=False,
        mode="img2img",
        enable_similar_image_filter=True,
        similar_image_filter_threshold=0.95,
    )

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    stream.stream.enable_similar_image_filter(threshold=0.95)

    for _ in range(stream.batch_size - 1):
        stream(image=sample_image)

    utilization_rec = []
    skip_probs = []

    for image_path in tqdm(images + [images[0]] * (stream.batch_size - 1)):
        output_image, skip_prob = stream(os.path.join(output, "frames", image_path))

        utilization_rec.append(torch.cuda.utilization())
        skip_probs.append(skip_prob)

    stream.stream.disable_similar_image_filter()

    for _ in range(stream.batch_size - 1):
        stream(image=sample_image)

    utilization_rec2 = []

    output_2 = os.path.join(output, "2")
    os.makedirs(output_2, exist_ok=True)

    for image_path in tqdm(images + [images[0]] * (stream.batch_size - 1)):
        output_image, skip_prob = stream(os.path.join(output, "frames", image_path))

        utilization_rec2.append(torch.cuda.utilization())

    # save fig of gpu usage
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax1 = plt.subplots()

    # Plotting the GPU utilization
    ax1.set_xlabel("Frame", fontsize=12)
    ax1.set_ylabel("GPU Utilization (%)", color="tab:blue", fontsize=12)
    ax1.plot(np.arange(len(utilization_rec)), utilization_rec, color="tab:blue", label="GPU Utilization 1")
    ax1.plot(np.arange(len(utilization_rec2)), utilization_rec2, color="tab:orange", label="GPU Utilization 2")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    # Creating a second y-axis for skip probability
    ax2 = ax1.twinx()
    ax2.set_ylabel("Skip Probability", color="tab:red", fontsize=12)
    ax2.plot(np.arange(len(skip_probs)), skip_probs, color="tab:red", label="Skip Probability")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    # Title and layout adjustments
    plt.title("GPU Processor Utilization and Skip Probability", fontsize=14)
    fig.tight_layout()  # Adjust the layout

    # Save the figure to a file
    plt.savefig(os.path.join("gpu_utilization.png"))


if __name__ == "__main__":
    fire.Fire(main)
