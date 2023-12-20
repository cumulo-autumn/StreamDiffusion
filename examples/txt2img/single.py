import os
import sys
from typing import Literal

import fire


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from wrapper import StreamDiffusionWrapper


def main(
    output: str = "output.png",
    model_id: str = "KBlueLeaf/kohaku-v2.1",
    LoRA_list: list = [],
    prompt: str = "Girl with panda ears wearing a hood",
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    seed: int = 2,
):
    stream = StreamDiffusionWrapper(
        model_id=model_id,
        LoRA_list = LoRA_list,
        t_index_list=[0, 16, 32, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        is_drawing=True,
        mode="txt2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type="none",
        seed = seed,
    )

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    for _ in range(stream.batch_size - 1):
        stream()

    output_image = stream()
    output_image.save(output)


if __name__ == "__main__":
    fire.Fire(main)
