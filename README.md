# ComfyUI-SkyCompute

ComfyUI nodes for the [Sky Compute](https://skycompute.ai) platform's services.
The first service is the **AI gateway**: prompts and images from hosted
models (OpenRouter today), with no API key in your workflow.

| Node | What it does |
|---|---|
| **Sky AI Gateway** | Where calls go. Optional: unconnected, the nodes use the workspace's gateway. |
| **Sky AI Prompt Generator** | Writes or improves a prompt with a language model. Attach images to have a vision model describe them or build on them. Outputs `prompt` and the model's `reasoning`. |
| **Sky AI Image Generator** | Makes images with a hosted image model: text to image, or guided by reference images. Outputs an `IMAGE` batch. |

The split follows [comfyui-ollama](https://github.com/stavsap/comfyui-ollama):
a connection node, and generators that take it as an optional input. One
difference: the connection does not pick the model. One gateway serves text
and image models, so each generator's **model** list shows the models it can
use (OpenRouter's `output_modalities`), with a refresh button.

Example: `example_workflows/sky-ai-prompt-to-image.json` (an idea, then a
prompt, then an image). It is in ComfyUI's template browser under
*ComfyUI-SkyCompute*.

## How it reaches models

In a Sky Compute workspace the pod runs `sky-ai-gateway`, an
OpenAI-compatible endpoint on `127.0.0.1:8098`, and sets
`SKY_AI_BASE_URL=http://127.0.0.1:8098/v1` in the app. The gateway holds
the providers' keys and logs each call's usage. ComfyUI never sees a key.

```
Sky AI node ──► $SKY_AI_BASE_URL  (sky-ai-gateway, in the pod) ──► OpenRouter
                 /v1/chat/completions   prompt generator
                 /v1/images             image generator (api: openrouter)
                 /v1/models             the model lists
```

Outside Sky Compute, set the same variables yourself, for any
OpenAI-compatible endpoint:

```bash
export SKY_AI_BASE_URL=https://openrouter.ai/api/v1
export SKY_AI_API_KEY=sk-or-...        # read from the environment only
```

`OPENAI_BASE_URL` and `OPENAI_API_KEY` are the fallback.

**Keys are never widgets.** Widget values are saved in the workflow and
in the metadata of every image it outputs. The key is read from the
environment and sent only to the environment's base URL. A **Sky AI
Gateway** node with another `base_url` calls it without a key, so a shared
workflow cannot send your key to another host.

## Install

In a Sky Compute ComfyUI workspace, open the terminal (or ComfyUI-Manager's
*Install via Git URL*):

```bash
cd "$SKY_ROOT/projects/$SKY_WORKSPACE/custom_nodes"
git clone https://github.com/DumbAI/ComfyUI-SkyCompute.git
```

Then restart ComfyUI (the workspace admin's **Restart**). The pack needs
only what ComfyUI ships (aiohttp, numpy, pillow and torch), and has no
`requirements.txt`.

Requires ComfyUI with the V3 node API (`comfy_api.latest`). Tested on
v0.45.6.

## The nodes

### Sky AI Gateway

| Input | |
|---|---|
| `base_url` | An OpenAI-compatible base URL ending in `/v1`. Empty means `$SKY_AI_BASE_URL`. |
| `timeout_s` | Seconds per call (default 300). |
| `max_retries` | Retries on 408/409/429/5xx and on connection errors, with backoff. *Advanced.* |

### Sky AI Prompt Generator

OpenAI chat completions (`POST /chat/completions`).

| Input | |
|---|---|
| `model` | A text model the gateway serves. |
| `prompt` | The idea, or any instruction. |
| `system` | Instructions to the model. The default writes text-to-image prompts. |
| `seed` | A new seed runs the node again. A seed other than 0 is sent to the model, and most models treat it as a hint. |
| `temperature`, `max_tokens` | *Advanced.* `max_tokens` 0 means the model's default. |
| `gateway`, `images` | Optional. Each image in the batch is sent to vision models as a PNG of at most 4 MP. |

The node shows its output text on the canvas.

### Sky AI Image Generator

| Input | |
|---|---|
| `model` | An image model the gateway serves. |
| `prompt` | What to draw, or the edit to make. Connect the prompt generator's `prompt` here. |
| `aspect_ratio`, `resolution` | `1:1` … `21:9`; `1K`/`2K`/`4K`. Models support different sets. |
| `count` | Images per run (1–4), returned as one batch. |
| `seed` | A new seed runs the node again. A seed other than 0 is sent to the models that take one. |
| `api` | The wire protocol. *Advanced.* `openrouter` is `POST /images` (the default; it takes reference images). `openai` is `POST /images/generations`, for a gateway route to an OpenAI-compatible image backend. |
| `gateway`, `images` | Optional. `images` are references for editing or guidance. |

## Develop

```
__init__.py              comfy_entrypoint (V3)
sky_compute/
  gateway.py             endpoint resolution, key rule, retries, errors      (plain Python)
  text.py                chat request and reply                              (plain Python)
  images.py              image protocols (APIS registry), decode, references (plain Python + PIL)
  models.py              model lists by output modality, cached 5 min       (plain Python)
  tensors.py             ComfyUI IMAGE tensors to and from PIL
  nodes.py               the V3 nodes
  routes.py              GET /skycompute/ai/models/{text|image}, the dropdowns' source
tests/                   the plain-Python modules against a fake gateway
example_workflows/       shown in ComfyUI's template browser
```

```bash
uv run pytest && uv run ruff check . && uv run ruff format --check .
```

To add an image protocol, write one class with `name`, `api`,
`supports_references` and `body()` in `images.py`, and add it to `APIS`.
It then appears in the node's `api` list.

The model dropdowns use ComfyUI's remote combo: the frontend reads a
JSON array from the route. The routes ask only the environment's gateway.
Since the options are not known when the schema is built, each generator's
`validate_inputs(model)` replaces ComfyUI's fixed-list check.

Publishing to the [Comfy Registry](https://registry.comfy.org): bump
`version` in `pyproject.toml`, then run the *Publish to Comfy Registry*
workflow. It needs the `REGISTRY_ACCESS_TOKEN` secret, created under the
`dumbai` publisher.

## License

GPL-3.0, like ComfyUI.
