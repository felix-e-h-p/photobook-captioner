"""Minimal Ollama client."""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Final

DEFAULT_HOST: Final = "http://localhost:11434"

type Message = dict[str, str]


class OllamaError(Exception):
    """Ollama couldn't be reached, or replied with something unusable."""


def chat(
    messages: list[Message],
    model: str,
    host: str | None = None,
    timeout: float = 30.0,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> str:
    """Send a chat to Ollama and return the reply text."""
    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": messages,
    }
    if json_mode:
        body["format"] = "json"
    host = host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
    request = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return str(json.loads(response.read())["message"]["content"])
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
        raise OllamaError(f"{type(e).__name__}: {e}") from e
