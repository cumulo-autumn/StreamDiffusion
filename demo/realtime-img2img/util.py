from importlib import import_module
from types import ModuleType
from typing import Dict, Any
from pydantic import BaseModel as PydanticBaseModel, Field
from PIL import Image
import io


def get_pipeline_class(pipeline_name: str) -> ModuleType:
    try:
        module = import_module(f"pipelines.{pipeline_name}")
    except ModuleNotFoundError:
        raise ValueError(f"Pipeline {pipeline_name} module not found")

    pipeline_class = getattr(module, "Pipeline", None)

    if pipeline_class is None:
        raise ValueError(f"'Pipeline' class not found in module '{pipeline_name}'.")

    return pipeline_class


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return image


def pil_to_frame(image: Image.Image) -> bytes:
    frame_data = io.BytesIO()
    image.save(frame_data, format="JPEG")
    frame_data = frame_data.getvalue()
    return (
        b"--frame\r\n"
        + b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
        + frame_data
        + b"\r\n"
    )


def is_firefox(user_agent: str) -> bool:
    return "Firefox" in user_agent
