import glob
import os
import sys
from typing import Literal

import fire

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from wrapper import StreamDiffusionWrapper


def main(
    input: str,
    output: str,
    model_id: str,
    prompt: str = "Girl with panda ears wearing a hood",
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "sfast", "tensorrt"] = "xformers",
):
    stream = StreamDiffusionWrapper(
        model_id=model_id,
        t_index_list=[32, 40, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        accerelation=acceleration,
        is_drawing=True,
    )

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    images = glob.glob(os.path.join(input, "*"))
    images = images + [images[-1]] * (stream.batch_size - 1)
    outputs = []

    for i in range(stream.batch_size - 1):
        image = images.pop(0)
        outputs.append(image)
        output_image = stream.img2img(image)
        output_image.save(os.path.join(output, f"{i}.png"))

    for image in images:
        outputs.append(image)
        try:
            output_image = stream.img2img(image)
        except Exception:
            continue

        name = outputs.pop(0)
        basename = os.path.basename(name)
        output_image.save(os.path.join(output, basename))


if __name__ == "__main__":
    fire.Fire(main)
