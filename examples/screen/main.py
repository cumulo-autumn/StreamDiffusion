import os
import sys
import time
import threading
from multiprocessing import Process, Queue, get_context
from multiprocessing.connection import Connection
from typing import List, Literal, Dict, Optional
import torch
import PIL.Image
from streamdiffusion.image_utils import pil2tensor
import mss
import fire
import tkinter as tk

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper

inputs = []
top = 0
left = 0

def screen(
    event: threading.Event,
    height: int = 512,
    width: int = 512,
    monitor: Dict[str, int] = {"top": 300, "left": 200, "width": 512, "height": 512},
):
    global inputs
    with mss.mss() as sct:
        while True:
            if event.is_set():
                print("terminate read thread")
                break
            img = sct.grab(monitor)
            img = PIL.Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
            img.resize((height, width))
            inputs.append(pil2tensor(img))
    print('exit : screen')
def dummy_screen(
        width: int,
        height: int,
):
    root = tk.Tk()
    root.title("Press Enter to start")
    root.geometry(f"{width}x{height}")
    root.resizable(False, False)
    root.attributes("-alpha", 0.8)
    root.configure(bg="black")
    def destroy(event):
        root.destroy()
    root.bind("<Return>", destroy)
    def update_geometry(event):
        global top, left
        top = root.winfo_y()
        left = root.winfo_x()
    root.bind("<Configure>", update_geometry)
    root.mainloop()
    return {"top": top, "left": left, "width": width, "height": height}

def monitor_setting_process(
    width: int,
    height: int,
    monitor_sender: Connection,
) -> None:
    monitor = dummy_screen(width, height)
    monitor_sender.send(monitor)

def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    close_queue: Queue,
    model_id_or_path: str,
    lora_dict: Optional[Dict[str, float]],
    prompt: str,
    negative_prompt: str,
    frame_buffer_size: int,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    use_denoising_batch: bool,
    seed: int,
    cfg_type: Literal["none", "full", "self", "initialize"],
    guidance_scale: float,
    delta: float,
    do_add_noise: bool,
    enable_similar_image_filter: bool,
    similar_image_filter_threshold: float,
    similar_image_filter_max_skip_frame: float,
    monitor_receiver : Connection,
) -> None:
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    queue : Queue
        The queue to put the generated images in.
    fps_queue : Queue
        The queue to put the calculated fps.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    negative_prompt : str, optional
        The negative prompt to use.
    frame_buffer_size : int, optional
        The frame buffer size for denoising batch, by default 1.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    cfg_type : Literal["none", "full", "self", "initialize"],
    optional
        The cfg_type for img2img mode, by default "self".
        You cannot use anything other than "none" for txt2img mode.
    guidance_scale : float, optional
        The CFG scale, by default 1.2.
    delta : float, optional
        The delta multiplier of virtual residual noise,
        by default 1.0.
    do_add_noise : bool, optional
        Whether to add noise for following denoising steps or not,
        by default True.
    enable_similar_image_filter : bool, optional
        Whether to enable similar image filter or not,
        by default False.
    similar_image_filter_threshold : float, optional
        The threshold for similar image filter, by default 0.98.
    similar_image_filter_max_skip_frame : int, optional
        The max skip frame for similar image filter, by default 10.
    """
    
    global inputs
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[32, 45],
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=similar_image_filter_threshold,
        similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
    )

    monitor = monitor_receiver.recv()

    event = threading.Event()
    input_screen = threading.Thread(target=screen, args=(event, height, width, monitor))
    input_screen.start()
    time.sleep(5)

    while True:
        try:
            if not close_queue.empty(): # closing check
                break
            if len(inputs) < frame_buffer_size:
                time.sleep(0.005)
                continue
            start_time = time.time()
            sampled_inputs = []
            for i in range(frame_buffer_size):
                index = (len(inputs) // frame_buffer_size) * i
                sampled_inputs.append(inputs[len(inputs) - index - 1])
            input_batch = torch.cat(sampled_inputs)
            inputs.clear()
            output_images = stream.stream(
                input_batch.to(device=stream.device, dtype=stream.dtype)
            ).cpu()
            if frame_buffer_size == 1:
                output_images = [output_images]
            for output_image in output_images:
                queue.put(output_image, block=False)

            fps = 1 / (time.time() - start_time)
            fps_queue.put(fps)
        except KeyboardInterrupt:
            break

    print("closing image_generation_process...")
    event.set() # stop capture thread
    input_screen.join()
    print(f"fps: {fps}")

def main(
    model_id_or_path: str = "KBlueLeaf/kohaku-v2.1",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    frame_buffer_size: int = 1,
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    seed: int = 2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    guidance_scale: float = 1.4,
    delta: float = 0.5,
    do_add_noise: bool = False,
    enable_similar_image_filter: bool = True,
    similar_image_filter_threshold: float = 0.99,
    similar_image_filter_max_skip_frame: float = 10,
) -> None:
    """
    Main function to start the image generation and viewer processes.
    """
    ctx = get_context('spawn')
    queue = ctx.Queue()
    fps_queue = ctx.Queue()
    close_queue = Queue()

    monitor_sender, monitor_receiver = ctx.Pipe()

    process1 = ctx.Process(
        target=image_generation_process,
        args=(
            queue,
            fps_queue,
            close_queue,
            model_id_or_path,
            lora_dict,
            prompt,
            negative_prompt,
            frame_buffer_size,
            width,
            height,
            acceleration,
            use_denoising_batch,
            seed,
            cfg_type,
            guidance_scale,
            delta,
            do_add_noise,
            enable_similar_image_filter,
            similar_image_filter_threshold,
            similar_image_filter_max_skip_frame,
            monitor_receiver,
            ),
    )
    process1.start()

    monitor_process = ctx.Process(
        target=monitor_setting_process,
        args=(
            width,
            height,
            monitor_sender,
            ),
    )
    monitor_process.start()
    monitor_process.join()

    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process2.start()

    # terminate
    process2.join()
    print("process2 terminated.")
    close_queue.put(True)
    print("process1 terminating...")
    process1.join(5) # with timeout
    if process1.is_alive():
        print("process1 still alive. force killing...")
        process1.terminate() # force kill...
    process1.join()
    print("process1 terminated.")


if __name__ == "__main__":
    fire.Fire(main)