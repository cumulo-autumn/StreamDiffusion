import threading
import time
import tkinter as tk
from multiprocessing import Process, Queue
from typing import List

import torch
from diffusers import (
    AutoencoderTiny,
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
)
from PIL import Image, ImageTk

from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
from streamdiffusion.image_utils import postprocess_image

image_update_counter = 0  # 画像の更新カウンター


def update_image(image_data: Image.Image, labels: List[tk.Label]) -> None:
    """
    Update the image displayed on a Tkinter label.

    Parameters
    ----------
    image_data : Image.Image
        The image to be displayed.
    labels : List[tk.Label]
        The list of labels where the image will be updated.
    """
    global image_update_counter
    label = labels[image_update_counter % len(labels)]
    image_update_counter += 1

    tk_image = ImageTk.PhotoImage(image_data)
    label.configure(image=tk_image)
    label.image = tk_image  # keep a reference


def image_generation_process(
    queue: Queue, prompt: str, model_name: str, batch_size: int = 10
) -> None:
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    queue : Queue
        The queue to put the generated images in.
    prompt : str
        The prompt to generate images from.
    model_name : str
        The name of the model to use for image generation.
    batch_size : int
        The batch size to use for image generation.
    """
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(model_name).to(
            device=torch.device("cuda"), dtype=torch.float16
        )
    except Exception:
        pipe = StableDiffusionPipeline.from_pretrained(model_name).to(
            device=torch.device("cuda")
        )

    denoising_steps = [0]
    stream = StreamDiffusion(
        pipe, denoising_steps, is_drawing=True, frame_buffer_size=batch_size
    )
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
        device=pipe.device, dtype=pipe.dtype
    )
    if model_name != "stabilityai/sd-turbo":
        stream.load_lcm_lora()
        stream.fuse_lora()

    # stream = accelerate_with_tensorrt(
    #     stream,
    #     f"./engines/{model_name}_max_batch_{len(denoising_steps)}",
    #     max_batch_size=len(denoising_steps),
    # )

    stream.prepare(prompt, num_inference_steps=50)

    main_thread_time_cumulative = 0.0
    lowpass_alpha = 0.1

    while True:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        x_outputs = stream.txt2img_batch(batch_size).cpu()
        queue.put(x_outputs)

        end.record()
        torch.cuda.synchronize()
        main_thread_time = start.elapsed_time(end) / 1000
        main_thread_time_cumulative = (
            lowpass_alpha * main_thread_time
            + (1 - lowpass_alpha) * main_thread_time_cumulative
        )
        fps = 1 / main_thread_time_cumulative * batch_size
        print(f"fps: {fps}, main_thread_time: {main_thread_time_cumulative}")


def _receive_images(queue: Queue, labels: List[tk.Label]) -> None:
    """
    Continuously receive images from a queue and update the labels.

    Parameters
    ----------
    queue : Queue
        The queue to receive images from.
    labels : List[tk.Label]
        The list of labels to update with images.
    """
    while True:
        if not queue.empty():
            images = postprocess_image(queue.get(), output_type="pil")
            for image_data in images:
                labels[0].after(0, update_image, image_data, labels)
        time.sleep(0.005)


def receive_images(queue: Queue) -> None:
    """
    Setup the Tkinter window and start the thread to receive images.

    Parameters
    ----------
    queue : Queue
        The queue to receive images from.
    """
    root = tk.Tk()
    root.title("Image Viewer")
    labels = [tk.Label(root) for _ in range(4)]
    for label in labels:
        label.pack(side=tk.LEFT)

    thread = threading.Thread(target=_receive_images, args=(queue, labels))
    thread.daemon = True
    thread.start()

    root.mainloop()


def main() -> None:
    """
    Main function to start the image generation and viewer processes.
    """
    queue = Queue()
    prompt = "cat with a hat, photoreal, 8K"
    model_name = "stabilityai/sd-turbo"
    batch_size = 10
    process1 = Process(
        target=image_generation_process, args=(queue, prompt, model_name, batch_size)
    )
    process1.start()

    process2 = Process(target=receive_images, args=(queue,))
    process2.start()


if __name__ == "__main__":
    main()
