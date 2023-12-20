import glob
import os
import sys
from typing import Literal, Dict, Optional

import fire


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    input: str = os.path.join(CURRENT_DIR, "..", "..", "images", "inputs"),
    output: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs"),
    model_id_or_path: str = "KBlueLeaf/kohaku-v2.1",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "1girl with panda ears wearing a hood",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    guidance_scale: float = 1.2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "initialize",
    seed: int = 2,
):
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    if guidance_scale <= 1.0:
        cfg_type = "none"

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[32, 40, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        is_drawing=True,
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
        cfg_type="self",
    )

    images = glob.glob(os.path.join(input, "*"))
    images = images + [images[-1]] * (stream.batch_size - 1)
    outputs = []

    for i in range(stream.batch_size - 1):
        image = images.pop(0)
        outputs.append(image)
        output_image = stream(image=image)
        output_image.save(os.path.join(output, f"{i}.png"))

    for image in images:
        outputs.append(image)
        try:
            output_image = stream(image=image)
        except Exception:
            continue

        name = outputs.pop(0)
        basename = os.path.basename(name)
        output_image.save(os.path.join(output, basename))


if __name__ == "__main__":
    fire.Fire(main)
