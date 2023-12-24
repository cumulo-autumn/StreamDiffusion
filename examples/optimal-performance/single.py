import os
import sys
import time
from multiprocessing import Process, Queue, get_context
from typing import Literal

import fire

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper

def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    prompt: str,
    model_id_or_path: str,
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
) -> None:
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    queue : Queue
        The queue to put the generated images in.
    fps_queue : Queue
        The queue to put the calculated fps.
    prompt : str
        The prompt to generate images from.
    model_id_or_path : str
        The name of the model to use for image generation.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    """
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        t_index_list=[0],
        frame_buffer_size=1,
        warmup=10,
        acceleration=acceleration,
        use_lcm_lora=False,
        mode="txt2img",
        cfg_type="none",
        use_denoising_batch=True,
    )

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    while True:
        try:
            start_time = time.time()

            x_outputs = stream.stream.txt2img_sd_turbo(1).cpu()
            queue.put(x_outputs, block=False)

            fps = 1 / (time.time() - start_time)
            fps_queue.put(fps)
        except KeyboardInterrupt:
            print(f"fps: {fps}")
            return

def main(
    prompt: str = "cat with sunglasses and a hat, photoreal, 8K",
    model_id_or_path: str = "stabilityai/sd-turbo",
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
) -> None:
    """
    Main function to start the image generation and viewer processes.
    """
    ctx = get_context('spawn')
    queue = ctx.Queue()
    fps_queue = ctx.Queue()
    process1 = ctx.Process(
        target=image_generation_process,
        args=(queue, fps_queue, prompt, model_id_or_path, acceleration),
    )
    process1.start()

    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process2.start()

    process1.join()
    process2.join()

if __name__ == "__main__":
    fire.Fire(main)
