import io
import os
import sys
import time
from multiprocessing import Process, Queue
from typing import List, Literal, Optional

import fire
import PIL.Image
import requests
import torch
from tqdm import tqdm

from streamdiffusion.image_utils import postprocess_image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from wrapper import StreamDiffusionWrapper


def _postprocess_image(queue: Queue) -> None:
    while True:
        try:
            if not queue.empty():
                output = postprocess_image(queue.get(block=False), output_type="pil")[0]
            time.sleep(0.0005)
        except KeyboardInterrupt:
            return


def download_image(url: str):
    response = requests.get(url)
    image = PIL.Image.open(io.BytesIO(response.content))
    return image


def run(
    warmup: int = 10,
    iterations: int = 100,
    model_id: str = "KBlueLeaf/kohaku-v2.1",
    LoRA_list: list = [],
    prompt: str = "Girl with brown dog ears,thick frame glasses",
    negative_prompt: str = "bad image , bad quality",
    use_lcm_lora: bool = True,
    use_tiny_vae: bool = True,
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    device_ids: Optional[List[int]] = None,
    use_denoising_batch: bool = True,
    seed: int = 2,
):
    stream = StreamDiffusionWrapper(
        model_id=model_id,
        LoRA_list = LoRA_list,
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
        enable_similar_image_filter=False,
        similar_image_filter_threshold=0.98,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type="self",  # initialize, full, self , none
        seed = seed,
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=1.2,
        delta=0.5,
    )

    image = download_image("https://github.com/ddpn08.png").resize((width, height))
    image_tensor = stream.preprocess_image(image)

    # warmup
    for _ in range(warmup):
        stream.stream(image_tensor)

    queue = Queue()
    p = Process(target=_postprocess_image, args=(queue,))
    p.start()

    results = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in tqdm(range(iterations)):
        start.record()
        out_tensor = stream.stream(image_tensor).cpu()
        queue.put(out_tensor)
        end.record()

        torch.cuda.synchronize()
        results.append(start.elapsed_time(end))

    print(f"Average time: {sum(results) / len(results)}ms")
    print(f"Average FPS: {1000 / (sum(results) / len(results))}")


if __name__ == "__main__":
    fire.Fire(run)
