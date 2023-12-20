import io
import os
import sys
from typing import List, Literal, Optional

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
    negative_prompt: str = "bad image , bad quality",
    use_lcm_lora: bool = True,
    use_tiny_vae: bool = True,
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "sfast", "tensorrt"] = "xformers",
    device_ids: Optional[List[int]] = None,
    use_denoising_batch: bool = True,
    seed: int = 2,
):
    stream = StreamDiffusionWrapper(
        model_id=model_id,
        use_lcm_lora=use_lcm_lora,
        use_tiny_vae=use_tiny_vae,
        t_index_list=[32, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=warmup,
        acceleration=acceleration,
        is_drawing=True,
        device_ids=device_ids,
        mode="img2img",
        use_denoising_batch = use_denoising_batch,
        cfg_type="initialize",  #initialize, full, self , none
        seed = seed,
    )

    stream.prepare(
        prompt = prompt,
        negative_prompt = negative_prompt,
        num_inference_steps=50,
        guidance_scale=1.4,
        delta=0.5,
    )

    downloaded_image = download_image("https://github.com/ddpn08.png").resize((width, height))
    
    # warmup
    for _ in range(warmup):
        image_tensor = stream.preprocess_image(downloaded_image)
        stream(image=image_tensor)

    results = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in tqdm(range(iterations)):
        start.record()
        image_tensor = stream.preprocess_image(downloaded_image)
        stream(image=image_tensor)
        end.record()

        torch.cuda.synchronize()
        results.append(start.elapsed_time(end))

    print(f"Average time: {sum(results) / len(results)}ms")
    print(f"Average FPS: {1000 / (sum(results) / len(results))}")
    import numpy as np
    fps_arr = 1000/np.array(results)
    print(f"Max FPS: {np.max(fps_arr)}")
    print(f"Min FPS: {np.min(fps_arr)}")
    print(f"Std: {np.std(fps_arr)}")


if __name__ == "__main__":
    fire.Fire(run)