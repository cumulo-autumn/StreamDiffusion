import os
import sys
from typing import Dict, List, Literal, Optional

from controlnet_aux.processor import Processor


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    output: str = os.path.join(
        CURRENT_DIR,
        "..",
        "..",
        "images",
        "outputs",
    ),
    model_id_or_path: str = "KBlueLeaf/kohaku-v2.1",
    lora_dict: Optional[Dict[str, float]] = None,
    controlnet_dicts: Optional[List[Dict[str, float]]] = None,
    prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    width: int = 512,
    height: int = 512,
    frame_buffer_size: int = 3,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    seed: int = 2,
):
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    output : str, optional
        The output image file to save images to.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    controlnet_dicts : Optional[List[Dict[str, float]], optional
        The controlnet_dicts to load, by default None.
        Keys are the ControlNet names and values are the ControlNet scales.
        Example: [{'controlnet_1' : 0.5}, {'controlnet_2' : 0.7},...]
    prompt : str
        The prompt to generate images from.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default False.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    """

    os.makedirs(output, exist_ok=True)

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        controlnet_dicts=controlnet_dicts,
        HyperSD_lora_id="Hyper-SD15-8steps-lora.safetensors",
        t_index_list=[0, 8, 16, 24, 32, 40, 45, 49],
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        acceleration=acceleration,
        CM_lora_type="Hyper_SD",
        mode="txt2img",
        use_denoising_batch=False,
        cfg_type="none",
        seed=seed,
    )

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    from PIL import Image

    controlnet_image = Image.open("/home/radius5/workspace/ono/StreamDiffusion/ookawakinketsu0379_TP_V.jpg")
    canny = Processor("canny")
    lineart = Processor("lineart_anime")
    controlnet_images = [canny(controlnet_image, to_pil=True), lineart(controlnet_image, to_pil=True)]

    output_images = stream(controlnet_images=controlnet_images)
    for i, output_image in enumerate(output_images):
        output_image.save(os.path.join(output, f"{i:02}.png"))


if __name__ == "__main__":
    # fire.Fire(main)
    main(
        output=os.path.join(
            CURRENT_DIR,
            "..",
            "..",
            "images",
            "outputs",
        ),
        model_id_or_path="KBlueLeaf/kohaku-v2.1",
        lora_dict=None,
        controlnet_dicts=[
            {"lllyasviel/control_v11p_sd15_canny": 1.0},
            {"lllyasviel/control_v11p_sd15s2_lineart_anime": 1.0},
        ],
        prompt="1girl with brown hair",
        width=512,
        height=512,
        frame_buffer_size=3,
        acceleration="xformers",
        seed=2,
    )
