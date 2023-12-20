import os
import sys
import threading
import time
import tkinter as tk
from multiprocessing import Process, Queue
from typing import List, Literal

import fire
from PIL import Image, ImageTk

from streamdiffusion.image_utils import postprocess_image


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper


def update_image(image_data: Image.Image, label: tk.Label) -> None:
    """
    Update the image displayed on a Tkinter label.

    Parameters
    ----------
    image_data : Image.Image
        The image to be displayed.
    label : tk.Label
        The labels where the image will be updated.
    """
    width = 512
    height = 512
    tk_image = ImageTk.PhotoImage(image_data, size=width)
    label.configure(image=tk_image, width=width, height=height)
    label.image = tk_image  # keep a reference


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


def _receive_images(
    queue: Queue, fps_queue: Queue, label: tk.Label, fps_label: tk.Label
) -> None:
    """
    Continuously receive images from a queue and update the labels.

    Parameters
    ----------
    queue : Queue
        The queue to receive images from.
    fps_queue : Queue
        The queue to put the calculated fps.
    label : tk.Label
        The label to update with images.
    fps_label : tk.Label
        The label to show fps.
    """
    while True:
        try:
            if not queue.empty():
                label.after(
                    0,
                    update_image,
                    postprocess_image(queue.get(block=False), output_type="pil")[0],
                    label,
                )
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
    label = tk.Label(root)
    fps_label = tk.Label(root, text="FPS: 0")
    label.grid(column=0)
    fps_label.grid(column=1)

    thread = threading.Thread(
        target=_receive_images, args=(queue, fps_queue, label, fps_label), daemon=True
    )
    thread.start()

    try:
        root.mainloop()
    except KeyboardInterrupt:
        return


def main(
    prompt: str = "cat with sunglasses and a hat, photoreal, 8K",
    model_id_or_path: str = "stabilityai/sd-turbo",
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
) -> None:
    """
    Main function to start the image generation and viewer processes.
    """
    queue = Queue()
    fps_queue = Queue()
    process1 = Process(
        target=image_generation_process,
        args=(queue, fps_queue, prompt, model_id_or_path, acceleration),
    )
    process1.start()

    process2 = Process(target=receive_images, args=(queue, fps_queue))
    process2.start()


if __name__ == "__main__":
    fire.Fire(main)
