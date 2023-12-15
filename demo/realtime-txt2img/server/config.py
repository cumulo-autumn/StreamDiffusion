from dataclasses import dataclass, field
from typing import List

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
    host: str = "0.0.0.0"
    port: int = 9090
    workers: int = 1

    ####################################################################
    # Model configuration
    ####################################################################
    # SD1.x variant model
    model_id: str = "KBlueLeaf/kohaku-v2.1"
    # LCM-LORA model
    lcm_lora_id: str = "latent-consistency/lcm-lora-sdv1-5"
    # TinyVAE model
    vae_id: str = "madebyollin/taesd"
    # Device to use
    device: torch.device = torch.device("cuda")
    # Data type
    dtype: torch.dtype = torch.float16

    ####################################################################
    # Inference configuration
    ####################################################################
    # Number of inference steps
    t_index_list: List[int] = field(default_factory=lambda: [0, 16, 32, 45])
    # Number of warmup steps
    warmup: int = 10
    safety_checker: bool = SAFETY_CHECKER
