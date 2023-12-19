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
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "sfast", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    guidance_scale: float = 1.2,
    delta: float = 1.0,
    cfg_type: Literal["none", "full", "self", "initialize"] = "initialize",
):
    negative_list = (
        [""]
        + [""] * 2
        + [negative_prompt] * 2
        + [""] * 4
        + [""] * 4
        + [negative_prompt] * 4
    )
    guidance_scale_list = (
        [1.0] + [1.2, 1.4] * 2 + [1.2, 1.2, 1.4, 1.4] * 2 + [1.2, 1.2, 1.4, 1.4] * 2
    )
    delta_list = [1.0] + [1.0] * 4 + [1.0, 0.5] * 2 + [1.0, 0.5] * 4
    cfg_type_list = ["none"] + ["full"] * 4 + ["self"] * 4 + ["initialize"] * 8

    for i in range(len(cfg_type_list)):
        negative_prompt = negative_list[i]
        guidance_scale = guidance_scale_list[i]
        delta = delta_list[i]
        cfg_type = cfg_type_list[i]

        file_name = ""
        if negative_prompt == "":
            file_name += "no_prompt"
        else:
            file_name += "with_prompt"

        file_name += "_gs_" + str(guidance_scale)
        file_name += "_del_" + str(delta)
        file_name += "_cfgtype_" + str(cfg_type)

        output = file_name + ".png"

        if guidance_scale <= 1.0:
            cfg_type = "none"

        stream = StreamDiffusionWrapper(
            model_id=model_id,
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
        )

        stream.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=guidance_scale,
            delta=delta,
        )

        image_tensor = stream.preprocess_image(input)

        for _ in range(stream.batch_size - 1):
            stream(image=image_tensor)

        output_image = stream(image=image_tensor)
        output_image.save(output)


if __name__ == "__main__":
    fire.Fire(main)
