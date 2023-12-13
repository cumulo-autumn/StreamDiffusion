import asyncio
import base64
import logging
from io import BytesIO
from pathlib import Path

import uvicorn
from config import Config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from PIL import Image
from pydantic import BaseModel
from wrapper import StreamDiffusionWrapper


logger = logging.getLogger("uvicorn")
PROJECT_DIR = Path(__file__).parent.parent


class PredictInputModel(BaseModel):
    """
    The input model for the /predict endpoint.
    """

    prompt: str


class PredictResponseModel(BaseModel):
    """
    The response model for the /predict endpoint.
    """

    base64_images: list[str]


class UpdatePromptResponseModel(BaseModel):
    """
    The response model for the /update_prompt endpoint.
    """

    prompt: str


class Api:
    def __init__(self, config: Config) -> None:
        """
        Initialize the API.

        Parameters
        ----------
        config : Config
            The configuration.
        """
        self.config = config
        self.stream_diffusion = StreamDiffusionWrapper(
            model_id=config.model_id,
            lcm_lora_id=config.lcm_lora_id,
            vae_id=config.vae_id,
            device=config.device,
            dtype=config.dtype,
            t_index_list=config.t_index_list,
            warmup=config.warmup,
            safety_checker=config.safety_checker,
        )
        self.app = FastAPI()
        self.app.add_api_route(
            "/api/predict",
            self._predict,
            methods=["POST"],
            response_model=PredictResponseModel,
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.mount(
            "/", StaticFiles(directory="../view/build", html=True), name="public"
        )

        self._predict_lock = asyncio.Lock()
        self._update_prompt_lock = asyncio.Lock()

        self.last_prompt: str = ""
        self.last_images: list[str] = [""]

    async def _predict(self, inp: PredictInputModel) -> PredictResponseModel:
        """
        Predict an image and return.

        Parameters
        ----------
        inp : PredictInputModel
            The input.

        Returns
        -------
        PredictResponseModel
            The prediction result.
        """
        async with self._predict_lock:
            if (
                self._calc_levenstein_distance(inp.prompt, self.last_prompt)
                < self.config.levenstein_distance_threshold
            ):
                logger.info("Using cached images")
                return PredictResponseModel(base64_images=self.last_images)
            self.last_prompt = inp.prompt
            self.last_images = [self._pil_to_base64(image) for image in self.stream_diffusion(inp.prompt)]
            return PredictResponseModel(base64_images=self.last_images)

    def _pil_to_base64(self, image: Image.Image, format: str = "JPEG") -> bytes:
        """
        Convert a PIL image to base64.

        Parameters
        ----------
        image : Image.Image
            The PIL image.

        format : str
            The image format, by default "JPEG".

        Returns
        -------
        bytes
            The base64 image.
        """
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("ascii")

    def _base64_to_pil(self, base64_image: str) -> Image.Image:
        """
        Convert a base64 image to PIL.

        Parameters
        ----------
        base64_image : str
            The base64 image.

        Returns
        -------
        Image.Image
            The PIL image.
        """
        if "base64," in base64_image:
            base64_image = base64_image.split("base64,")[1]
        return Image.open(BytesIO(base64.b64decode(base64_image))).convert("RGB")

    def _calc_levenstein_distance(self, a: str, b: str) -> int:
        """
        Calculate the Levenstein distance between two strings.

        Parameters
        ----------
        a : str
            The first string.

        b : str
            The second string.

        Returns
        -------
        int
            The Levenstein distance.
        """
        if a == b:
            return 0
        a_k = len(a)
        b_k = len(b)
        if a == "":
            return b_k
        if b == "":
            return a_k
        matrix = [[] for i in range(a_k + 1)]
        for i in range(a_k + 1):
            matrix[i] = [0 for j in range(b_k + 1)]
        for i in range(a_k + 1):
            matrix[i][0] = i
        for j in range(b_k + 1):
            matrix[0][j] = j
        for i in range(1, a_k + 1):
            ac = a[i - 1]
            for j in range(1, b_k + 1):
                bc = b[j - 1]
                cost = 0 if (ac == bc) else 1
                matrix[i][j] = min(
                    [
                        matrix[i - 1][j] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j - 1] + cost,
                    ]
                )
        return matrix[a_k][b_k]


if __name__ == "__main__":
    from config import Config

    config = Config()

    uvicorn.run(
        Api(config).app,
        host=config.host,
        port=config.port,
        workers=config.workers,
    )
