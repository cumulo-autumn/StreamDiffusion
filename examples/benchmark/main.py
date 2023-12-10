import os
import sys
import io
from typing import *

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
    acceleration: Literal["none", "xformers", "sfast", "tensorrt"] = "none",
    device_ids: Optional[List[int]] = None,
):
    stream = StreamDiffusionWrapper(
        model_id=model_id,
        use_lcm_lora=use_lcm_lora,
        use_tiny_vae=use_tiny_vae,
        t_index_list=[35, 45],
        frame_buffer_size=1,
        warmup=warmup,
        accerelation=acceleration,
        is_drawing=True,
        device_ids=device_ids,
    )

    stream.prepare(
        prompt,
        num_inference_steps=50,
    )

    image = download_image("https://github.com/ddpn08.png").resize((512, 512))

    # warmup
    for _ in range(warmup):
        stream.img2img(image)

    results = []

    for _ in tqdm(range(iterations)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        stream.img2img(image)
        end.record()

        torch.cuda.synchronize()
        results.append(start.elapsed_time(end))

    print(f"Average time: {sum(results) / len(results)}ms")
    print(f"Average FPS: {1000 / (sum(results) / len(results))}")


if __name__ == "__main__":
    fire.Fire(run)
