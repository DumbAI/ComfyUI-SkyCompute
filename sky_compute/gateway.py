"""The client for the Sky Compute AI gateway, or any OpenAI-compatible endpoint.

In a Sky Compute workspace the pod runs `sky-ai-gateway` on loopback and sets
`SKY_AI_BASE_URL` (http://127.0.0.1:8098/v1). It holds the providers' keys and
needs none from the caller. Elsewhere, point `SKY_AI_BASE_URL` at any
OpenAI-compatible endpoint (OpenRouter, for one) and put its key in
`SKY_AI_API_KEY`.

Keys come from the environment only, never from a node's widgets: widgets are
saved into the workflow and the PNG metadata of every output image. The key is
sent only to the endpoint the environment names. A workflow that overrides the
base URL talks to it without one, so a shared workflow cannot point the key at
another host.

No ComfyUI imports here: this module is unit-tested on its own.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import aiohttp

log = logging.getLogger("sky_compute")

BASE_URL_VARS = ("SKY_AI_BASE_URL", "OPENAI_BASE_URL")
API_KEY_VARS = ("SKY_AI_API_KEY", "OPENAI_API_KEY")

# Statuses worth another try: rate limits and an unreachable or overloaded
# upstream. The gateway answers 502 when it cannot reach a provider.
RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


class GatewayError(RuntimeError):
    """A failed call, with the message the endpoint gave (OpenAI's error shape)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class NotConfigured(GatewayError):
    pass


@dataclass(frozen=True)
class Gateway:
    """Where calls go: the value a Sky AI Gateway node outputs.

    `base_url` empty means the environment's endpoint. Holds no key, so it is
    safe to pass between nodes and to cache.
    """

    base_url: str = ""
    timeout_s: int = 300
    max_retries: int = 2

    def resolve(self, environ: Mapping[str, str] | None = None) -> tuple[str, str | None]:
        """The base URL to call and the key to send with it (None: no key)."""
        environ = os.environ if environ is None else environ
        configured = next((environ[v] for v in BASE_URL_VARS if environ.get(v)), "")
        url = (self.base_url or configured).strip().rstrip("/")
        if not url:
            raise NotConfigured(
                "No AI gateway: SKY_AI_BASE_URL is not set. In a Sky Compute workspace the "
                "platform sets it when models are configured; elsewhere set it (and "
                "SKY_AI_API_KEY) or give the Sky AI Gateway node a base URL."
            )
        if not url.startswith(("http://", "https://")):
            raise GatewayError(f"The gateway URL must be http(s): {url!r}")
        own = configured.strip().rstrip("/")
        key = next((environ[v] for v in API_KEY_VARS if environ.get(v)), None)
        return url, key if url == own else None


DEFAULT = Gateway()


async def call(
    gateway: Gateway,
    method: str,
    api: str,
    *,
    body: dict[str, Any] | None = None,
    params: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
    session: aiohttp.ClientSession | None = None,
) -> dict[str, Any]:
    """One JSON call to `<base>/<api>`, retried on transient failures."""
    base, key = gateway.resolve(environ)
    url = f"{base}/{api.lstrip('/')}"
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    timeout = aiohttp.ClientTimeout(total=gateway.timeout_s, sock_connect=30)

    own = session is None
    session = session or aiohttp.ClientSession(timeout=timeout)
    try:
        for attempt in range(gateway.max_retries + 1):
            last = attempt == gateway.max_retries
            try:
                async with session.request(
                    method, url, json=body, params=params, headers=headers, timeout=timeout
                ) as r:
                    text = await r.text()
                    if r.status < 400:
                        return _json(text, url)
                    error = GatewayError(_message(text, r.status), r.status)
                    if last or r.status not in RETRY_STATUSES:
                        raise error
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                if last:
                    raise GatewayError(f"Could not reach the AI gateway at {base}: {e or type(e).__name__}") from e
                error = e
            delay = min(2**attempt, 8)
            log.warning("sky_compute: %s %s failed (%s), retrying in %ss", method, api, error, delay)
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")
    finally:
        if own:
            await session.close()


async def fetch_bytes(url: str, timeout_s: int = 120) -> bytes:
    """Download an http(s) URL a provider returned (a generated image)."""
    if not url.startswith(("http://", "https://")):
        raise GatewayError(f"Not a downloadable URL: {url[:80]!r}")
    timeout = aiohttp.ClientTimeout(total=timeout_s, sock_connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as s, s.get(url) as r:
        if r.status >= 400:
            raise GatewayError(f"Downloading {url[:80]} failed: HTTP {r.status}", r.status)
        return await r.read()


def _json(text: str, url: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except ValueError as e:
        raise GatewayError(f"{url} answered with something other than JSON: {text[:200]!r}") from e
    if not isinstance(data, dict):
        raise GatewayError(f"{url} answered with JSON that is not an object")
    if isinstance(data.get("error"), dict):  # some providers report errors with a 200
        raise GatewayError(_message(text, 200))
    return data


def _message(text: str, status: int) -> str:
    """The error's message, from OpenAI's shape when the body has it."""
    try:
        error = json.loads(text).get("error")
    except (ValueError, AttributeError):
        error = None
    if isinstance(error, dict) and error.get("message"):
        detail = error["message"]
        raw = (error.get("metadata") or {}).get("raw") if isinstance(error.get("metadata"), dict) else None
        return f"HTTP {status}: {detail}" + (f" ({raw})" if raw else "")
    return f"HTTP {status}: {text[:300] or 'no body'}"
