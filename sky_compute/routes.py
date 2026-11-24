"""ComfyUI server routes: the model lists behind the nodes' model dropdowns.

`GET /skycompute/ai/models/{text|image}` answers a JSON array of model ids
(the shape the frontend's remote combo widgets read). It asks only the
environment's gateway: a request cannot name another host.
"""

from __future__ import annotations

import logging

from aiohttp import web

from . import gateway as gw
from .models import KINDS, list_models

log = logging.getLogger("sky_compute")

MODELS_ROUTE = "/skycompute/ai/models"


async def models_handler(request: web.Request) -> web.Response:
    kind = request.match_info["kind"]
    if kind not in KINDS:
        return web.json_response({"error": f"kind must be one of {list(KINDS)}"}, status=404)
    try:
        return web.json_response(await list_models(kind))
    except gw.GatewayError as e:
        log.warning("sky_compute: listing %s models failed: %s", kind, e)
        return web.json_response({"error": str(e)}, status=502)


def register() -> None:
    """Add the routes to ComfyUI's server; called once, while custom nodes load."""
    from server import PromptServer  # ComfyUI's; imported here so the rest of the package loads without it

    PromptServer.instance.routes.get(MODELS_ROUTE + "/{kind}")(models_handler)
