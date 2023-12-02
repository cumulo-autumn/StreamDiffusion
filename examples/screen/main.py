import io
import multiprocessing as mp
import threading
import time
from time import sleep
from typing import *

import fire
import mss
import PIL.Image
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from matplotlib import pyplot as plt
from socks import UDP, receive_udp_data

from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
from streamdiffusion.image_utils import pil2tensor, postprocess_image


input = []


def screen(
    height: int = 512,
    width: int = 512,
    monitor: Dict[str, int] = {"top": 300, "left": 200, "width": 512, "height": 512},
):
    global input
    with mss.mss() as sct:
        while True:
            img = sct.grab(monitor)
            img = PIL.Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
            img.resize((height, width))
            input.append(pil2tensor(img))


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


def run(prompt: str = "Girl with panda ears wearing a hood", 
        address: str = "127.0.0.1", 
        port: int = 8080, 
        frame_buffer_size: int = 1):
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file("./model.safetensors").to(
        device=torch.device("cuda")
    )

    stream = StreamDiffusion(
        pipe,
        [32, 45],
        frame_buffer_size = frame_buffer_size,
    )
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
    stream.load_lcm_lora()
    stream.fuse_lora()
    stream.enable_similar_image_filter(0.95)
    stream = accelerate_with_tensorrt(stream, "./engines", max_batch_size=6)
    stream.prepare(
        prompt,
        num_inference_steps=50,
    )

    output_window = mp.Process(target=result_window, args=(address, port))
    input_screen = threading.Thread(target=screen)

    output_window.start()
    print("Waiting for output window to start...")
    time.sleep(5)
    input_screen.start()

    udp = UDP(address, port)

    main_thread_time_cumulative = 0
    lowpass_alpha = 0.1

    while True:
        if len(input) < frame_buffer_size:
            sleep(0.01)
            continue

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        
        input_batch = torch.cat(input[:frame_buffer_size])
        x_output = stream(input_batch.to(device=stream.device, dtype=stream.dtype))
        output_images = postprocess_image(x_output, output_type="pil")

        for output_image in output_images:
            udp.send_udp_data(output_image)
        end.record()
        torch.cuda.synchronize()
        main_thread_time = start.elapsed_time(end) / (1000 * output_images[0].shape[0])
        main_thread_time_cumulative = (
            lowpass_alpha * main_thread_time + (1 - lowpass_alpha) * main_thread_time_cumulative
        )
        fps = 1 / main_thread_time_cumulative
        print(f"fps: {fps}, main_thread_time: {main_thread_time_cumulative}")


if __name__ == "__main__":
    fire.Fire(run)
