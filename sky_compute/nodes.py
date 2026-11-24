"""The nodes (ComfyUI's V3 API): a gateway connection, a prompt generator and an image generator.

They mirror comfyui-ollama's split: a connection node says where calls go, and
the generators take it as an optional input. Unlike Ollama's, the connection
does not pick the model: one gateway serves text and image models, so each
generator lists the models it can use.
"""

from __future__ import annotations

import logging

from comfy_api.latest import ComfyExtension, io, ui

from . import gateway as gw
from . import images as img
from . import text
from .routes import MODELS_ROUTE
from .tensors import from_pils, to_pils

log = logging.getLogger("sky_compute")

CATEGORY = "Sky Compute/AI"
GATEWAY = io.Custom("SKY_AI_GATEWAY")
SEED_MAX = 2**31 - 1


def _model_input(kind: str, tooltip: str) -> io.Combo.Input:
    # Filled from the gateway when the node is added and on refresh; the
    # node's validate_inputs takes over the list check (the options are not
    # known when the schema is built).
    return io.Combo.Input(
        "model",
        options=[],
        remote=io.RemoteOptions(route=f"{MODELS_ROUTE}/{kind}", refresh_button=True),
        tooltip=tooltip,
    )


def _gateway_input() -> io.Input:
    return GATEWAY.Input(
        "gateway",
        optional=True,
        tooltip="From a Sky AI Gateway node. Unconnected: the workspace's gateway ($SKY_AI_BASE_URL).",
    )


def _check_model(model: str) -> bool | str:
    return True if isinstance(model, str) and model.strip() else "Pick a model (use the refresh button to list them)."


class SkyAIGateway(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SkyCompute_AIGateway",
            display_name="Sky AI Gateway",
            category=CATEGORY,
            description=(
                "Where the Sky AI nodes send their calls: the workspace's AI gateway, an "
                "OpenAI-compatible endpoint that holds the providers' keys. Optional: the "
                "generators use the workspace's gateway when nothing is connected."
            ),
            search_aliases=["openai", "openrouter", "llm connection"],
            inputs=[
                io.String.Input(
                    "base_url",
                    default="",
                    placeholder="the workspace's gateway ($SKY_AI_BASE_URL)",
                    tooltip=(
                        "An OpenAI-compatible base URL, ending in /v1. Empty: the workspace's "
                        "gateway. API keys are never set here; a URL other than the environment's "
                        "is called without one."
                    ),
                ),
                io.Int.Input("timeout_s", default=300, min=5, max=3600, tooltip="Seconds per call."),
                io.Int.Input(
                    "max_retries",
                    default=2,
                    min=0,
                    max=5,
                    tooltip="Retries on rate limits and unreachable providers.",
                    advanced=True,
                ),
            ],
            outputs=[GATEWAY.Output(display_name="gateway")],
        )

    @classmethod
    def validate_inputs(cls, base_url: str) -> bool | str:
        url = base_url.strip()
        return True if not url or url.startswith(("http://", "https://")) else "base_url must start with http(s)://"

    @classmethod
    def execute(cls, base_url: str, timeout_s: int, max_retries: int) -> io.NodeOutput:
        return io.NodeOutput(gw.Gateway(base_url=base_url.strip(), timeout_s=timeout_s, max_retries=max_retries))


class SkyAIPromptGenerator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SkyCompute_AIPromptGenerator",
            display_name="Sky AI Prompt Generator",
            category=CATEGORY,
            description=(
                "Writes or improves a prompt with a language model through the AI gateway "
                "(OpenAI chat completions). Attach images to describe or build on them."
            ),
            search_aliases=["llm", "prompt enhance", "chat", "caption", "vision"],
            inputs=[
                _model_input("text", "A text model the gateway serves."),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="a lighthouse on a cliff at dusk",
                    tooltip="The idea to turn into a prompt, or any instruction.",
                ),
                io.String.Input(
                    "system",
                    multiline=True,
                    default=text.PROMPT_ENGINEER,
                    tooltip="The model's instructions. The default writes text-to-image prompts.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=SEED_MAX,
                    control_after_generate=True,
                    tooltip="A new seed runs the node again. Sent to the model when not 0; many treat it as a hint.",
                ),
                io.Float.Input(
                    "temperature",
                    default=0.8,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Higher is more varied.",
                    advanced=True,
                ),
                io.Int.Input(
                    "max_tokens",
                    default=0,
                    min=0,
                    max=65536,
                    tooltip="The reply's length limit. 0: the model's default.",
                    advanced=True,
                ),
                _gateway_input(),
                io.Image.Input("images", optional=True, tooltip="Images for a vision model to look at."),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="reasoning", tooltip="The model's thinking, when it shares it."),
            ],
        )

    @classmethod
    def validate_inputs(cls, model: str) -> bool | str:
        return _check_model(model)

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        system: str,
        seed: int,
        temperature: float,
        max_tokens: int,
        gateway: gw.Gateway | None = None,
        images=None,
    ) -> io.NodeOutput:
        if not prompt.strip():
            raise ValueError("The prompt is empty.")
        refs = []
        if images is not None:
            refs = [img.to_data_url(p) for p in to_pils(images)]
        req = text.ChatRequest(
            model=model,
            prompt=prompt,
            system=system,
            images=refs,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        reply = text.parse_reply(await gw.call(gateway or gw.DEFAULT, "POST", text.API, body=req.body()))
        log.info("sky_compute: %s wrote %d characters, usage %s", model, len(reply.text), reply.usage)
        return io.NodeOutput(reply.text, reply.reasoning, ui=ui.PreviewText(reply.text))


class SkyAIImageGenerator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SkyCompute_AIImageGenerator",
            display_name="Sky AI Image Generator",
            category=CATEGORY,
            description=(
                "Generates images with a hosted image model through the AI gateway. Attach "
                "reference images to edit or guide (OpenRouter's image API)."
            ),
            search_aliases=["text to image", "openrouter image", "image edit"],
            inputs=[
                _model_input("image", "An image model the gateway serves."),
                io.String.Input("prompt", multiline=True, default="", tooltip="What to draw, or the edit to make."),
                io.Combo.Input("aspect_ratio", options=img.ASPECT_RATIOS, default="1:1"),
                io.Combo.Input(
                    "resolution",
                    options=img.RESOLUTIONS,
                    default="1K",
                    tooltip="Size tier: 1K is about one megapixel. Models support different tiers.",
                ),
                io.Int.Input("count", default=1, min=1, max=4, tooltip="Images per run, one batch."),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=SEED_MAX,
                    control_after_generate=True,
                    tooltip="A new seed runs the node again. Sent when not 0, for the models that take one.",
                ),
                io.Combo.Input(
                    "api",
                    options=list(img.APIS),
                    default="openrouter",
                    tooltip=(
                        "The wire protocol: openrouter is POST /images (the platform's default "
                        "backend); openai is POST /images/generations."
                    ),
                    advanced=True,
                ),
                _gateway_input(),
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Reference images for editing or guidance; scaled to at most 4 megapixels.",
                ),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def validate_inputs(cls, model: str) -> bool | str:
        return _check_model(model)

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
        count: int,
        seed: int,
        api: str,
        gateway: gw.Gateway | None = None,
        images=None,
    ) -> io.NodeOutput:
        g = gateway or gw.DEFAULT
        protocol = img.APIS[api]
        req = img.ImageRequest(
            model=model,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            count=count,
            seed=seed,
            references=[img.to_data_url(p) for p in to_pils(images)] if images is not None else [],
        )
        img.validate(protocol, req)
        data = await gw.call(g, "POST", protocol.api, body=protocol.body(req))
        pils = await img.decode_reply(data, lambda url: gw.fetch_bytes(url, g.timeout_s))
        log.info("sky_compute: %s made %d image(s), usage %s", model, len(pils), data.get("usage"))
        return io.NodeOutput(from_pils(pils))


NODES: list[type[io.ComfyNode]] = [SkyAIGateway, SkyAIPromptGenerator, SkyAIImageGenerator]


class SkyComputeExtension(ComfyExtension):
    async def on_load(self) -> None:
        from .routes import register

        register()

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODES
