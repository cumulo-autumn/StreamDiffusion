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

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

image_update_counter = 0


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

    tk_image = ImageTk.PhotoImage(image_data, size=512)
    label.configure(image=tk_image)
    label.image = tk_image  # keep a reference


def image_generation_process(
    queue: Queue, fps_queue: Queue, prompt: str, model_name: str, batch_size: int = 10
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

    stream = accelerate_with_tensorrt(
        stream,
        f"./engines/{model_name}_max_batch_{batch_size}_min_batch_{batch_size}",
        max_batch_size=batch_size,
        min_batch_size=batch_size,
    )

    stream.prepare(prompt, num_inference_steps=50)

    while True:
        try:
            start_time = time.time()

            x_outputs = stream.txt2img_batch(batch_size).cpu()
            queue.put(x_outputs, block=False)

            main_thread_time_cumulative = time.time() - start_time
            fps = 1 / main_thread_time_cumulative * batch_size
            fps_queue.put(fps)
        except KeyboardInterrupt:
            print(f"fps: {fps}, main_thread_time: {main_thread_time_cumulative}")
            break


def _receive_images(queue: Queue, fps_queue: Queue, labels: List[tk.Label], fps_label: tk.Label) -> None:
    """
    Continuously receive images from a queue and update the labels.

    Parameters
    ----------
    queue : Queue
        The queue to receive images from.
    fps_queue : Queue
        The queue to put the calculated fps.
    labels : List[tk.Label]
        The list of labels to update with images.
    fps_label : tk.Label
        The label to show fps.
    """
    while True:
        try:
            if not queue.empty():
                [
                    labels[0].after(0, update_image, image_data, labels)
                    for image_data in postprocess_image(queue.get(block=False), output_type="pil")
                ]
            if not fps_queue.empty():
                fps_label.config(text=f"FPS: {fps_queue.get(block=False):.2f}")

            time.sleep(0.0005)
        except KeyboardInterrupt:
            break


def receive_images(queue: Queue, fps_queue: Queue) -> None:
    """
    Setup the Tkinter window and start the thread to receive images.

    Parameters
    ----------
    queue : Queue
        The queue to receive images from.
    fps_queue : Queue
        The queue to put the calculated fps.
    """
    root = tk.Tk()
    root.title("Image Viewer")
    labels = [tk.Label(root) for _ in range(4)]
    labels[0].grid(row=0, column=0)
    labels[1].grid(row=0, column=1)
    labels[2].grid(row=1, column=0)
    labels[3].grid(row=1, column=1)
    fps_label = tk.Label(root, text="FPS: 0")
    fps_label.grid(rows=2, columnspan=2)

    thread = threading.Thread(target=_receive_images, args=(queue, fps_queue, labels, fps_label), daemon=True)
    thread.start()

    root.mainloop()


def main() -> None:
    """
    Main function to start the image generation and viewer processes.
    """
    queue = Queue()
    fps_queue = Queue()
    prompt = "cat with a hat, photoreal, 8K"
    model_name = "stabilityai/sd-turbo"
    batch_size = 12
    process1 = Process(
        target=image_generation_process, args=(queue, fps_queue, prompt, model_name, batch_size)
    )
    process1.start()

    process2 = Process(target=receive_images, args=(queue, fps_queue))
    process2.start()


if __name__ == "__main__":
    main()
