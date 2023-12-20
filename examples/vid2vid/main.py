import os
import sys
from typing import Literal, Dict, Optional

import fire
import torch
from torchvision.io import read_video, write_video
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    input: str,
    output: str = os.path.join(
        CURRENT_DIR, "..", "..", "images", "outputs", "output.mp4"
    ),
    model_id: str = "KBlueLeaf/kohaku-v2.1",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "1girl with brown dog ears, thick frame glasses",
    scale: float = 1.0,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    enable_similar_image_filter: bool = True,
    seed: int = 2,
):
    video_info = read_video(input)
    video = video_info[0] / 255
    fps = video_info[2]["video_fps"]
    width = int(video.shape[1] * scale)
    height = int(video.shape[2] * scale)

    stream = StreamDiffusionWrapper(
        model_id=model_id,
        lora_dict=lora_dict,
        t_index_list=[35, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=False,
        mode="img2img",
        output_type="pt",
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=0.98,
        use_denoising_batch=use_denoising_batch,
        seed=seed,
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
    write_video(output, video_result, fps=fps)


if __name__ == "__main__":
    fire.Fire(main)
