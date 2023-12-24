import os
import sys
import threading
import time
import tkinter as tk
from multiprocessing import Process, Queue, get_context
from typing import List, Literal

import fire
from PIL import Image, ImageTk

from streamdiffusion.image_utils import postprocess_image


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper


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

    width = 320
    height = 320
    tk_image = ImageTk.PhotoImage(image_data.resize((width, height)), size=width)
    label.configure(image=tk_image, width=width, height=height)
    label.image = tk_image  # keep a reference


def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    prompt: str,
    model_id_or_path: str,
    batch_size: int = 10,
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
    batch_size : int
        The batch size to use for image generation.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    """
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        t_index_list=[0],
        frame_buffer_size=batch_size,
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

            x_outputs = stream.stream.txt2img_sd_turbo(batch_size).cpu()
            queue.put(x_outputs, block=False)

            fps = 1 / (time.time() - start_time) * batch_size
            fps_queue.put(fps)
        except KeyboardInterrupt:
            print(f"fps: {fps}")
            return


def _receive_images(
    queue: Queue, fps_queue: Queue, labels: List[tk.Label], fps_label: tk.Label
) -> None:
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
                    for image_data in postprocess_image(
                        queue.get(block=False), output_type="pil"
                    )
                ]
            if not fps_queue.empty():
                fps_label.config(text=f"FPS: {fps_queue.get(block=False):.2f}")

            time.sleep(0.0005)
        except KeyboardInterrupt:
            return


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

    thread = threading.Thread(
        target=_receive_images, args=(queue, fps_queue, labels, fps_label), daemon=True
    )
    thread.start()

    try:
        root.mainloop()
    except KeyboardInterrupt:
        return


def main(
    prompt: str = "cat with sunglasses and a hat, photoreal, 8K",
    model_id_or_path: str = "stabilityai/sd-turbo",
    batch_size: int = 12,
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
        args=(queue, fps_queue, prompt, model_id_or_path, batch_size, acceleration),
    )
    process1.start()

    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process2.start()

    process1.join()
    process2.join()

if __name__ == "__main__":
    fire.Fire(main)
