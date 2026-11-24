"""The gateway's models, by what they output: the nodes' model dropdowns."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from . import gateway as gw

KINDS = ("text", "image")
TTL_S = 300

_cache: dict[tuple[str, str], tuple[float, list[str]]] = {}


async def list_models(
    kind: str,
    g: gw.Gateway = gw.DEFAULT,
    *,
    environ: Mapping[str, str] | None = None,
    now: float | None = None,
) -> list[str]:
    """Model ids that output `kind`, sorted, cached for five minutes per endpoint.

    Asks with OpenRouter's `output_modalities` filter and filters again on each
    model's `architecture`, since other OpenAI-compatible endpoints ignore the
    query. A model without that field is listed under every kind.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    url, _ = g.resolve(environ)
    now = time.monotonic() if now is None else now
    hit = _cache.get((url, kind))
    if hit and now - hit[0] < TTL_S:
        return hit[1]
    data = await gw.call(g, "GET", "models", params={"output_modalities": kind}, environ=environ)
    ids = sorted({m["id"] for m in data.get("data") or [] if isinstance(m, dict) and m.get("id") and _outputs(m, kind)})
    _cache[(url, kind)] = (now, ids)
    return ids


def _outputs(model: dict[str, Any], kind: str) -> bool:
    modalities = (model.get("architecture") or {}).get("output_modalities")
    return not isinstance(modalities, list) or kind in modalities
