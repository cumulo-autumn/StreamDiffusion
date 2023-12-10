import os
import sys
from typing import Literal

import fire

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from wrapper import StreamDiffusionWrapper


def main(
    input: str,
    output: str = "output.png",
    model_id: str = "KBlueLeaf/kohaku-v2.1",
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
        mode="img2img",
    )

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    image_tensor = stream.preprocess_image(input)

    for _ in range(stream.batch_size - 1):
        stream(image=image_tensor)

    output_image = stream(image=image_tensor)
    output_image.save(output)


if __name__ == "__main__":
    fire.Fire(main)
