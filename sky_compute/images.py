"""Image generation: one request, several wire protocols.

An `ImageApi` turns an `ImageRequest` into one provider protocol's body and
reads that protocol's reply. The gateway routes each API path to a backend
plugin, so choosing the protocol here chooses the path:

- `openrouter`: OpenRouter's `POST /images` (aspect ratio, resolution tiers,
  reference images, seed). The default, since the platform's backend is
  OpenRouter.
- `openai`: OpenAI's `POST /images/generations` (pixel sizes, no references),
  for a gateway route to an OpenAI-compatible image backend.

Adding a protocol is one class and one entry in `APIS`. Pixels are PIL here;
the nodes convert to and from ComfyUI's tensors.
"""

from __future__ import annotations

import base64
import binascii
import io
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from PIL import Image

from .gateway import GatewayError

ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"]
RESOLUTIONS = ["1K", "2K", "4K"]
REFERENCE_MAX_PIXELS = 2048 * 2048


@dataclass(frozen=True)
class ImageRequest:
    model: str
    prompt: str
    aspect_ratio: str = "1:1"
    resolution: str = "1K"
    count: int = 1
    seed: int = 0  # 0: none
    references: list[str] = field(default_factory=list)  # data: URLs


class ImageApi(Protocol):
    name: ClassVar[str]
    api: ClassVar[str]  # the path under the gateway's /v1
    supports_references: ClassVar[bool]

    def body(self, req: ImageRequest) -> dict[str, Any]: ...


class OpenRouterImages:
    name = "openrouter"
    api = "images"
    supports_references = True

    def body(self, req: ImageRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": req.model,
            "prompt": req.prompt,
            "aspect_ratio": req.aspect_ratio,
            "resolution": req.resolution,
            "n": req.count,
            # Inline: OpenRouter's URLs expire, and the gateway must not have to be reachable twice.
            "response_format": "b64_json",
        }
        if req.seed > 0:
            body["seed"] = req.seed
        if req.references:
            body["input_references"] = [{"type": "image_url", "image_url": {"url": u}} for u in req.references]
        return body


class OpenAIImages:
    name = "openai"
    api = "images/generations"
    supports_references = False

    def body(self, req: ImageRequest) -> dict[str, Any]:
        # gpt-image models take three sizes; pick by orientation. They always answer b64_json.
        w, h = (int(x) for x in req.aspect_ratio.split(":"))
        size = "1024x1024" if w == h else ("1536x1024" if w > h else "1024x1536")
        return {"model": req.model, "prompt": req.prompt, "n": req.count, "size": size}


APIS: dict[str, ImageApi] = {a.name: a for a in (OpenRouterImages(), OpenAIImages())}


def validate(api: ImageApi, req: ImageRequest) -> None:
    if not req.prompt.strip():
        raise GatewayError("The prompt is empty.")
    if req.references and not api.supports_references:
        raise GatewayError(f"The {api.name} image API takes no reference images; use the openrouter API.")
    if req.aspect_ratio not in ASPECT_RATIOS:
        raise GatewayError(f"Unknown aspect ratio {req.aspect_ratio!r}.")


async def decode_reply(data: dict[str, Any], download: Callable[[str], Awaitable[bytes]]) -> list[Image.Image]:
    """The images of an `/images`-style reply: `data[]` with `b64_json` or a `url`."""
    images = []
    for item in data.get("data") or []:
        if item.get("b64_json"):
            raw = _b64(item["b64_json"])
        elif item.get("url"):
            url = item["url"]
            raw = _data_url(url) if url.startswith("data:") else await download(url)
        else:
            continue
        images.append(_open(raw))
    if not images:
        raise GatewayError("The model returned no image.")
    return same_size(images)


def same_size(images: list[Image.Image]) -> list[Image.Image]:
    """One batch needs one size and mode: the first image's."""
    first = images[0]
    return [first] + [
        (im if im.size == first.size else im.resize(first.size, Image.Resampling.LANCZOS)).convert(first.mode)
        for im in images[1:]
    ]


def to_data_url(image: Image.Image, max_pixels: int = REFERENCE_MAX_PIXELS) -> str:
    """A reference image as a PNG data URL, scaled down to at most `max_pixels`."""
    w, h = image.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _open(raw: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as e:
        raise GatewayError(f"The model returned an image that could not be decoded: {e}") from e
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    return image.convert("RGBA" if has_alpha else "RGB")


def _data_url(url: str) -> bytes:
    header, _, payload = url.partition(",")
    if ";base64" not in header:
        raise GatewayError("The model returned a data URL that is not base64.")
    return _b64(payload)


def _b64(s: str) -> bytes:
    try:
        return base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError) as e:
        raise GatewayError(f"The model returned invalid base64: {e}") from e
