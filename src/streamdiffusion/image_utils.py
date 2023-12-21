from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torchvision


def denormalize(images: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Denormalize an image array to [0,1].
    """
    return (images / 2 + 0.5).clamp(0, 1)


def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images


def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
    """
    Convert a NumPy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [
            PIL.Image.fromarray(image.squeeze(), mode="L") for image in images
        ]
    else:
        pil_images = [PIL.Image.fromarray(image) for image in images]

    return pil_images


def postprocess_image(
    image: torch.Tensor,
    output_type: str = "pil",
    do_denormalize: Optional[List[bool]] = None,
) -> Union[torch.Tensor, np.ndarray, PIL.Image.Image]:
    if not isinstance(image, torch.Tensor):
        raise ValueError(
            f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
        )

    if output_type == "latent":
        return image

    do_normalize_flg = True
    if do_denormalize is None:
        do_denormalize = [do_normalize_flg] * image.shape[0]

    image = torch.stack(
        [
            denormalize(image[i]) if do_denormalize[i] else image[i]
            for i in range(image.shape[0])
        ]
    )

    if output_type == "pt":
        return image

    image = pt_to_numpy(image)

    if output_type == "np":
        return image

    if output_type == "pil":
        return numpy_to_pil(image)


def process_image(
    image_pil: PIL.Image.Image, range: Tuple[int, int] = (-1, 1)
) -> Tuple[torch.Tensor, PIL.Image.Image]:
    image = torchvision.transforms.ToTensor()(image_pil)
    r_min, r_max = range[0], range[1]
    image = image * (r_max - r_min) + r_min
    return image[None, ...], image_pil


def pil2tensor(image_pil: PIL.Image.Image) -> torch.Tensor:
    height = image_pil.height
    width = image_pil.width
    imgs = []
    img, _ = process_image(image_pil)
    imgs.append(img)
    imgs = torch.vstack(imgs)
    images = torch.nn.functional.interpolate(
        imgs, size=(height, width), mode="bilinear"
    )
    image_tensors = images.to(torch.float16)
    return image_tensors
