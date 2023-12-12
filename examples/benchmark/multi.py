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

import uuid
def _postprocess_image(queue: Queue) -> None:
    while True:
        try:
            if not queue.empty():
                output = postprocess_image(queue.get(block=False), output_type="pil")[0]
                output.save(f"{str(uuid.uuid4())}.png")
            time.sleep(0.0005)
        except KeyboardInterrupt:
            break

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
    use_denoising_batch: bool = True,
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
        acceleration=acceleration,
        is_drawing=True,
        device_ids=device_ids,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
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
        stream.stream(image_tensor)

    queue = Queue()
    p = Process(target=_postprocess_image, args=(queue,))
    p.start()

    results = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in tqdm(range(iterations)):
        start.record()
        out_tensor = stream.stream(image_tensor)
        queue.put(out_tensor)
        end.record()

        torch.cuda.synchronize()
        results.append(start.elapsed_time(end))

    print(f"Average time: {sum(results) / len(results)}ms")
    print(f"Average FPS: {1000 / (sum(results) / len(results))}")


if __name__ == "__main__":
    fire.Fire(run)