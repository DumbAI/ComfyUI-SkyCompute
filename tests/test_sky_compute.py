"""The plain-Python core against a fake OpenAI-compatible gateway. No ComfyUI needed."""

from __future__ import annotations

import base64
import io

import pytest
from aiohttp import web
from PIL import Image

from sky_compute import gateway as gw
from sky_compute import images as img
from sky_compute import models, text
from sky_compute.routes import models_handler


def png(color=(255, 0, 0), size=(8, 4)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


# --- endpoint resolution: where calls go, and when a key goes with them ---


def test_the_environment_names_the_gateway_and_its_key():
    env = {"SKY_AI_BASE_URL": "http://127.0.0.1:8098/v1/", "SKY_AI_API_KEY": "k"}
    assert gw.DEFAULT.resolve(env) == ("http://127.0.0.1:8098/v1", "k")


def test_openai_variables_are_the_fallback():
    env = {"OPENAI_BASE_URL": "https://openrouter.ai/api/v1", "OPENAI_API_KEY": "sk"}
    assert gw.DEFAULT.resolve(env) == ("https://openrouter.ai/api/v1", "sk")


def test_a_url_from_the_workflow_never_gets_the_key():
    env = {"SKY_AI_BASE_URL": "http://127.0.0.1:8098/v1", "SKY_AI_API_KEY": "k"}
    assert gw.Gateway(base_url="https://evil.example/v1").resolve(env) == ("https://evil.example/v1", None)
    # ...unless it is the environment's own.
    assert gw.Gateway(base_url="http://127.0.0.1:8098/v1/").resolve(env)[1] == "k"


def test_no_gateway_is_a_clear_error():
    with pytest.raises(gw.NotConfigured, match="SKY_AI_BASE_URL"):
        gw.DEFAULT.resolve({})
    with pytest.raises(gw.GatewayError, match="http"):
        gw.Gateway(base_url="ftp://x").resolve({})


# --- requests and replies (pure) ---


def test_a_chat_request_carries_images_as_parts_and_only_set_options():
    body = text.ChatRequest(model="m", prompt="p", system="s", images=["data:x"], seed=7).body()
    assert body["messages"][0] == {"role": "system", "content": "s"}
    assert body["messages"][1]["content"] == [
        {"type": "text", "text": "p"},
        {"type": "image_url", "image_url": {"url": "data:x"}},
    ]
    assert body["seed"] == 7 and "max_tokens" not in body and "temperature" not in body
    plain = text.ChatRequest(model="m", prompt="p").body()
    assert plain["messages"] == [{"role": "user", "content": "p"}]


def test_a_chat_reply_gives_text_and_reasoning():
    reply = text.parse_reply(
        {"choices": [{"message": {"content": " a prompt \n", "reasoning": "hmm"}}], "usage": {"total_tokens": 3}}
    )
    assert (reply.text, reply.reasoning, reply.usage) == ("a prompt", "hmm", {"total_tokens": 3})
    parts = text.parse_reply({"choices": [{"message": {"content": [{"type": "text", "text": "x"}]}}]})
    assert parts.text == "x"
    with pytest.raises(gw.GatewayError, match="finish_reason: length"):
        text.parse_reply({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})


def test_image_protocols_shape_their_bodies():
    req = img.ImageRequest(model="m", prompt="p", aspect_ratio="16:9", count=2, seed=5, references=["data:r"])
    body = img.APIS["openrouter"].body(req)
    assert body["aspect_ratio"] == "16:9" and body["resolution"] == "1K" and body["n"] == 2 and body["seed"] == 5
    assert body["input_references"] == [{"type": "image_url", "image_url": {"url": "data:r"}}]
    assert body["response_format"] == "b64_json"
    assert img.APIS["openai"].body(req)["size"] == "1536x1024"
    assert img.APIS["openai"].body(img.ImageRequest(model="m", prompt="p", aspect_ratio="2:3"))["size"] == "1024x1536"
    with pytest.raises(gw.GatewayError, match="no reference images"):
        img.validate(img.APIS["openai"], req)
    with pytest.raises(gw.GatewayError, match="empty"):
        img.validate(img.APIS["openrouter"], img.ImageRequest(model="m", prompt=" "))


async def test_image_replies_decode_b64_data_urls_and_downloads_into_one_size():
    b64 = base64.b64encode(png(size=(8, 4))).decode()
    downloads = []

    async def download(url):
        downloads.append(url)
        return png((0, 0, 255), size=(16, 8))

    out = await img.decode_reply(
        {"data": [{"b64_json": b64}, {"url": "data:image/png;base64," + b64}, {"url": "https://cdn/x.png"}]},
        download,
    )
    assert downloads == ["https://cdn/x.png"]
    assert [im.size for im in out] == [(8, 4)] * 3
    with pytest.raises(gw.GatewayError, match="no image"):
        await img.decode_reply({"data": []}, download)
    with pytest.raises(gw.GatewayError, match="could not be decoded"):
        await img.decode_reply({"data": [{"b64_json": "bm90IGFuIGltYWdl"}]}, download)


def test_references_are_scaled_down_png_data_urls():
    url = img.to_data_url(Image.new("RGB", (400, 100)), max_pixels=100 * 25)
    assert url.startswith("data:image/png;base64,")
    assert Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size == (100, 25)


# --- against a fake gateway ---


@pytest.fixture
async def fake(aiohttp_server):
    seen: list[web.Request] = []
    state = {"fail": 0}

    async def chat(request):
        seen.append(request)
        if state["fail"]:
            state["fail"] -= 1
            return web.json_response({"error": {"message": "upstream down", "type": "x"}}, status=502)
        return web.json_response({"choices": [{"message": {"content": "a red fox"}}]})

    async def bad(request):
        return web.json_response({"error": {"message": "no such model", "metadata": {"raw": "m?"}}}, status=400)

    async def models_(request):
        seen.append(request)
        return web.json_response(
            {
                "data": [
                    {"id": "b/img", "architecture": {"output_modalities": ["image", "text"]}},
                    {"id": "a/txt", "architecture": {"output_modalities": ["text"]}},
                    {"id": "c/plain"},
                ]
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/bad", bad)
    app.router.add_get("/v1/models", models_)
    server = await aiohttp_server(app)
    return {"url": str(server.make_url("/v1")), "seen": seen, "state": state}


async def test_a_call_sends_the_environments_key_to_the_environments_gateway(fake):
    env = {"SKY_AI_BASE_URL": fake["url"], "SKY_AI_API_KEY": "k"}
    data = await gw.call(gw.DEFAULT, "POST", "chat/completions", body={"model": "m"}, environ=env)
    assert data["choices"][0]["message"]["content"] == "a red fox"
    assert fake["seen"][-1].headers["Authorization"] == "Bearer k"


async def test_transient_failures_are_retried_and_errors_keep_the_providers_message(fake, monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(gw.asyncio, "sleep", no_sleep)
    env = {"SKY_AI_BASE_URL": fake["url"]}
    fake["state"]["fail"] = 2
    await gw.call(gw.Gateway(max_retries=2), "POST", "chat/completions", body={}, environ=env)
    assert len(fake["seen"]) == 3
    fake["state"]["fail"] = 1
    with pytest.raises(gw.GatewayError, match="upstream down") as e:
        await gw.call(gw.Gateway(max_retries=0), "POST", "chat/completions", body={}, environ=env)
    assert e.value.status == 502
    with pytest.raises(gw.GatewayError, match=r"HTTP 400: no such model \(m\?\)"):
        await gw.call(gw.DEFAULT, "POST", "bad", body={}, environ=env)


async def test_an_unreachable_gateway_names_itself(unused_tcp_port):
    env = {"SKY_AI_BASE_URL": f"http://127.0.0.1:{unused_tcp_port}/v1"}
    with pytest.raises(gw.GatewayError, match="Could not reach the AI gateway"):
        await gw.call(gw.Gateway(max_retries=0), "GET", "models", environ=env)


async def test_models_are_listed_by_output_and_cached(fake):
    models._cache.clear()
    env = {"SKY_AI_BASE_URL": fake["url"]}
    assert await models.list_models("image", environ=env, now=0) == ["b/img", "c/plain"]
    assert fake["seen"][-1].query["output_modalities"] == "image"
    assert await models.list_models("text", environ=env, now=0) == ["a/txt", "b/img", "c/plain"]
    calls = len(fake["seen"])
    await models.list_models("text", environ=env, now=models.TTL_S - 1)
    assert len(fake["seen"]) == calls
    await models.list_models("text", environ=env, now=models.TTL_S + 1)
    assert len(fake["seen"]) == calls + 1


async def test_the_models_route_answers_an_array_or_a_502(aiohttp_client, fake, monkeypatch):
    models._cache.clear()
    app = web.Application()
    app.router.add_get("/skycompute/ai/models/{kind}", models_handler)
    client = await aiohttp_client(app)
    monkeypatch.setenv("SKY_AI_BASE_URL", fake["url"])
    r = await client.get("/skycompute/ai/models/text")
    assert r.status == 200 and await r.json() == ["a/txt", "b/img", "c/plain"]
    assert (await client.get("/skycompute/ai/models/audio")).status == 404
    monkeypatch.delenv("SKY_AI_BASE_URL")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    r = await client.get("/skycompute/ai/models/image")
    assert r.status == 502 and "SKY_AI_BASE_URL" in (await r.json())["error"]
