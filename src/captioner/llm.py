"""Optional LLM writer that rewrites page facts into a more natural caption.

Runs a small open model (Llama 3.2 3B) locally through Ollama.
Free, no API key is needed and nothing leaves the machine.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Final

from .facts import PageFacts
from .writers import TemplateWriter, Writer, WriterError

DEFAULT_MODEL: Final = "llama3.2:3b"
DEFAULT_HOST: Final = "http://localhost:11434"

SYSTEM_PROMPT: Final = """You write short titles for pages in a printed photo book.
You are given facts about the photos on one page. Write ONE title.

Rules:
- Short - under 60 characters. Warm and natural, the way a person would title their own photos.
- Only use the place and date given. Never invent names, places, dates or details.
- Always keep the year. To make it shorter, drop the country or shorten the month
  (e.g. "Feb 2019") - never drop the year.
- Don't say who is in the photos (no "solo", "friends", "grandma") - only mention kids or
  family if the facts say so.
- If a fact is null, leave it out. You may leave the place out if the title reads better.
- No emojis, no quotation marks, no full stop at the end.
Reply with the title only."""

EXAMPLES: Final[list[tuple[dict[str, Any], str]]] = [
    (
        {"occasion": "Beach day", "place": "Brighton", "date": "July 2023", "family": True},
        "Family day at the beach in Brighton, July 2023",
    ),
    (
        {"occasion": "BBQ", "place": None, "date": "12th June 2025", "setting": "outdoor"},
        "12th June 2025 – BBQ in the garden",
    ),
    (
        {"occasion": "Skiing holiday", "place": "Morzine, France", "date": "February 2024"},
        "Ski week in Morzine, Feb 2024",
    ),
    (
        {"occasion": None, "place": "Whitby", "date": "August 2022", "season": "summer"},
        "Summer days in Whitby, August 2022",
    ),
]


def facts_for_prompt(f: PageFacts) -> dict[str, Any]:
    """The only thing the model sees. A few words of text, nothing beyond what goes on
    the printed page anyway."""
    return {
        "occasion": f.phrase,
        "place": f.place if f.use_place else None,
        "date": f.date_text,
        "season": f.season,
        "setting": f.setting,
        "kids": f.kids,
        "family": f.family,
        "photos_on_page": f.n_photos,
    }


def _clean(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line.strip().strip("\"'“”").rstrip(".").strip()


class LLMWriter:
    """Local LLM writer via Ollama. Raises WriterError on any failure so the pipeline
    can fall back to the template.

    Model and host can be changed with CAPTION_LLM_MODEL and OLLAMA_HOST.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: float = 30.0,
        temperature: float = 0.3,
    ) -> None:
        self.model = model or os.environ.get("CAPTION_LLM_MODEL", DEFAULT_MODEL)
        self.host = host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
        self.timeout = timeout
        self.temperature = temperature
        self.name = f"llm:{self.model}"

    def messages(self, facts: PageFacts) -> list[dict[str, str]]:
        """System prompt, worked examples, then this page's facts."""
        out = [{"role": "system", "content": SYSTEM_PROMPT}]
        for example, title in EXAMPLES:
            out.append({"role": "user", "content": json.dumps(example)})
            out.append({"role": "assistant", "content": title})
        out.append({"role": "user", "content": json.dumps(facts_for_prompt(facts))})
        return out

    def complete(self, messages: list[dict[str, str]]) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": messages,
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return str(json.loads(response.read())["message"]["content"])
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
            raise WriterError(f"{type(e).__name__}: {e}") from e

    def write(self, facts: PageFacts) -> str:
        caption = _clean(self.complete(self.messages(facts)))
        if not caption:
            raise WriterError("model returned an empty caption")
        return caption


def writer_from_env() -> Writer:
    """CAPTION_WRITER=template (default) or llm."""
    match os.environ.get("CAPTION_WRITER", "template"):
        case "template":
            return TemplateWriter()
        case "llm":
            return LLMWriter()
        case other:
            raise ValueError(f"CAPTION_WRITER must be 'template' or 'llm', got {other!r}")
