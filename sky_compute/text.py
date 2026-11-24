"""Text generation over OpenAI's chat completions API: the prompt generator's request and reply."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gateway import GatewayError

API = "chat/completions"

PROMPT_ENGINEER = (
    "You write prompts for text-to-image models. Turn the user's idea into one "
    "vivid, self-contained prompt: subject, composition, setting, lighting, style "
    "and medium, in natural language. If images are attached, ground the prompt in "
    "them. Reply with the prompt only: no preamble, no quotes, no markdown."
)


@dataclass(frozen=True)
class ChatRequest:
    model: str
    prompt: str
    system: str = ""
    images: list[str] = field(default_factory=list)  # data: URLs
    temperature: float | None = None
    max_tokens: int = 0  # 0: the provider's default
    seed: int = 0  # 0: none

    def body(self) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if self.system.strip():
            messages.append({"role": "system", "content": self.system})
        if self.images:
            content: Any = [{"type": "text", "text": self.prompt}] + [
                {"type": "image_url", "image_url": {"url": url}} for url in self.images
            ]
        else:
            content = self.prompt
        messages.append({"role": "user", "content": content})
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens > 0:
            body["max_tokens"] = self.max_tokens
        if self.seed > 0:
            body["seed"] = self.seed
        return body


@dataclass(frozen=True)
class ChatReply:
    text: str
    reasoning: str
    usage: dict[str, Any] | None


def parse_reply(data: dict[str, Any]) -> ChatReply:
    choices = data.get("choices") or []
    if not choices:
        raise GatewayError("The model returned no choices.")
    message = choices[0].get("message") or {}
    text = _text(message.get("content"))
    # OpenRouter puts thinking in `reasoning`; DeepSeek-style providers in `reasoning_content`.
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    if not text.strip():
        finish = choices[0].get("finish_reason")
        raise GatewayError(
            "The model returned no text"
            + (f" (finish_reason: {finish})" if finish else "")
            + ". Try a higher max_tokens or another model."
        )
    return ChatReply(text=text.strip(), reasoning=str(reasoning).strip(), usage=data.get("usage"))


def _text(content: Any) -> str:
    """Content is a string, or (some providers) a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""
