import io
import multiprocessing as mp
import os
import sys
import threading
import time
from time import sleep
from typing import Dict, Literal

import fire
import mss
import PIL.Image
import torch
from matplotlib import pyplot as plt
from socks import UDP, receive_udp_data

from streamdiffusion.image_utils import pil2tensor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from wrapper import StreamDiffusionWrapper

inputs = []


def screen(
    height: int = 512,
    width: int = 512,
    monitor: Dict[str, int] = {"top": 300, "left": 200, "width": 512, "height": 512},
):
    global inputs
    with mss.mss() as sct:
        while True:
            img = sct.grab(monitor)
            img = PIL.Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
            img.resize((height, width))
            inputs.append(pil2tensor(img))


def result_window(server_ip: str, server_port: int):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    while True:
        received_data = receive_udp_data(server_ip, server_port)
        images = PIL.Image.open(io.BytesIO(received_data))
        ax.clear()
        ax.imshow(images)
        ax.axis("off")
        plt.pause(0.00001)


def run(
    model_id: str = "KBlueLeaf/kohaku-v2.1",
    prompt: str = "Girl with brown dog ears,thick frame glasses",
    negative_prompt: str = "bad image , bad quality",
    address: str = "127.0.0.1",
    port: int = 8080,
    frame_buffer_size: int = 1,
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "sfast", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
):
    stream = StreamDiffusionWrapper(
        model_id=model_id,
        t_index_list=[32, 45],
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        is_drawing=False,
        enable_similar_image_filter=True,
        similar_image_filter_threshold=0.98,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type="self",  # initialize, full, self
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=1.4,
        delta=0.5,
    )

    output_window = mp.Process(target=result_window, args=(address, port))
    input_screen = threading.Thread(target=screen, args=(height, width))

    output_window.start()
    print("Waiting for output window to start...")
    time.sleep(5)
    input_screen.start()

    udp = UDP(address, port)

    main_thread_time_cumulative = 0
    lowpass_alpha = 0.1

    while True:
        if len(inputs) < frame_buffer_size:
            sleep(0.005)
            continue

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        sampled_inputs = []
        for i in range(frame_buffer_size):
            index = (len(inputs) // frame_buffer_size) * i
            sampled_inputs.append(inputs[len(inputs) - index - 1])

        input_batch = torch.cat(sampled_inputs)
        inputs.clear()
        output_images = stream(
            image=input_batch.to(device=stream.device, dtype=stream.dtype)
        )

        if frame_buffer_size == 1:
            output_images = [output_images]
        for output_image in output_images:
            udp.send_udp_data(output_image)
        end.record()
        torch.cuda.synchronize()
        main_thread_time = start.elapsed_time(end) / (1000 * frame_buffer_size)
        main_thread_time_cumulative = (
            lowpass_alpha * main_thread_time
            + (1 - lowpass_alpha) * main_thread_time_cumulative
        )
        fps = 1 / main_thread_time_cumulative
        print(f"fps: {fps}, main_thread_time: {main_thread_time_cumulative}")


if __name__ == "__main__":
    fire.Fire(run)
