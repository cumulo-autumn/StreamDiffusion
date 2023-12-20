import os
import sys
from typing import Literal, Dict, Optional

import fire

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    output: str = os.path.join(
        CURRENT_DIR, "..", "..", "images", "outputs", "output.png"
    ),
    model_id: str = "KBlueLeaf/kohaku-v2.1",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "Girl with panda ears wearing a hood",
    width: int = 512,
    height: int = 512,
    frame_buffer_size: int = 3,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    seed: int = 2,
):
    os.makedirs(output, exist_ok=True)

    stream = StreamDiffusionWrapper(
        model_id=model_id,
        lora_dict=lora_dict,
        t_index_list=[0, 16, 32, 45],
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        is_drawing=True,
        mode="txt2img",
        use_denoising_batch=False,
        cfg_type="none",
        seed=seed,
    )

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    output_images = stream()
    for i, output_image in enumerate(output_images):
        output_image.save(os.path.join(output, f"{i:02}.png"))


if __name__ == "__main__":
    fire.Fire(main)
