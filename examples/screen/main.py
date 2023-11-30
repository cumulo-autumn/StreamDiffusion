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
from diffusers import AutoencoderTiny, LCMScheduler, StableDiffusionPipeline
from matplotlib import pyplot as plt
from socks import UDP, receive_udp_data

from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
from streamdiffusion.image_utils import pil2tensor, postprocess_image


input = None


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
            input = pil2tensor(img)


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


def run(address: str = "127.0.0.1", port: int = 8080):
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file("./model.safetensors").to(
        device=torch.device("cuda")
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()

    stream = StreamDiffusion(
        pipe,
        [32, 45],
    )
    stream = accelerate_with_tensorrt(stream, "./engines", max_batch_size=2)
    stream.prepare(
        "Girl with panda ears wearing a hood",
        num_inference_steps=50,
        generator=torch.manual_seed(2),
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
        if input is None:
            sleep(0.01)
            continue

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        x_output = stream(input.to(device=stream.device, dtype=stream.dtype))
        output_images = postprocess_image(x_output, output_type="pil")[0]

        udp.send_udp_data(output_images)
        end.record()
        torch.cuda.synchronize()
        main_thread_time = start.elapsed_time(end) / 1000
        main_thread_time_cumulative = (
            lowpass_alpha * main_thread_time + (1 - lowpass_alpha) * main_thread_time_cumulative
        )
        fps = 1 / main_thread_time_cumulative
        print(f"fps: {fps}, main_thread_time: {main_thread_time_cumulative}")


if __name__ == "__main__":
    fire.Fire(run)
