from dataclasses import dataclass, field
from typing import List, Literal

import torch
import os

SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", "False") == "True"


@dataclass
class Config:
    """
    The configuration for the API.
    """

    ####################################################################
    # Server
    ####################################################################
    # In most cases, you should leave this as it is.
    host: str = "127.0.0.1"
    port: int = 9090
    workers: int = 1

    ####################################################################
    # Model configuration
    ####################################################################
    mode: Literal["txt2img", "img2img"] = "txt2img"
    # SD1.x variant model
    model_id_or_path: str = os.environ.get("MODEL", "KBlueLeaf/kohaku-v2.1")
    # LoRA dictionary write like    field(default_factory=lambda: {'E:/stable-diffusion-webui/models/Lora_1.safetensors' : 1.0 , 'E:/stable-diffusion-webui/models/Lora_2.safetensors' : 0.2})
    lora_dict: dict = None
    # LCM-LORA model
    lcm_lora_id: str = os.environ.get("LORA", "latent-consistency/lcm-lora-sdv1-5")
    # TinyVAE model
    vae_id: str = os.environ.get("VAE", "madebyollin/taesd")
    # Device to use
    device: torch.device = torch.device("cuda")
    # Data type
    dtype: torch.dtype = torch.float16
    # acceleration
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers"

    ####################################################################
    # Inference configuration
    ####################################################################
    # Number of inference steps
    t_index_list: List[int] = field(default_factory=lambda: [0, 16, 32, 45])
    # Number of warmup steps
    warmup: int = 10
    use_safety_checker: bool = SAFETY_CHECKER
