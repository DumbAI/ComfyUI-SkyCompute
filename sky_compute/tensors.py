"""ComfyUI's IMAGE tensors ([batch, height, width, channel], float in 0..1) to and from PIL."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def to_pils(images: torch.Tensor) -> list[Image.Image]:
    out = []
    for image in images:
        array = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
        out.append(Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array))
    return out


def from_pils(images: list[Image.Image]) -> torch.Tensor:
    """A batch of RGB images: downstream nodes expect three channels."""
    arrays = [np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0 for im in images]
    return torch.from_numpy(np.stack(arrays))
