import os
import sys
import io
from typing import Literal, List, Optional

import fire
import PIL.Image
import requests
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from wrapper import StreamDiffusionWrapper


def download_image(url: str):
    response = requests.get(url)
    image = PIL.Image.open(io.BytesIO(response.content))
    return image


def run(
    warmup: int = 10,
    iterations: int = 100,
    model_id: str = "KBlueLeaf/kohaku-v2.1",
    prompt: str = "Girl with panda ears wearing a hood",
    use_lcm_lora: bool = True,
    use_tiny_vae: bool = True,
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "sfast", "tensorrt"] = "xformers",
    device_ids: Optional[List[int]] = None,
):
    stream = StreamDiffusionWrapper(
        model_id=model_id,
        use_lcm_lora=use_lcm_lora,
        use_tiny_vae=use_tiny_vae,
        t_index_list=[35, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=warmup,
        accerelation=acceleration,
        is_drawing=True,
        device_ids=device_ids,
    )

    stream.prepare(
        prompt,
        num_inference_steps=50,
    )

    image_tensor = stream.preprocess_image(
        download_image("https://github.com/ddpn08.png").resize((width, height))
    )

    # warmup
    for _ in range(warmup):
        stream.img2img(image_tensor)

    results = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in tqdm(range(iterations)):
        start.record()
        stream.img2img(image_tensor)
        end.record()

        torch.cuda.synchronize()
        results.append(start.elapsed_time(end))

    print(f"Average time: {sum(results) / len(results)}ms")
    print(f"Average FPS: {1000 / (sum(results) / len(results))}")


if __name__ == "__main__":
    fire.Fire(run)
