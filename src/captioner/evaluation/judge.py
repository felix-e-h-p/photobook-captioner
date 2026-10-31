"""LLM-as-judge - a second, larger local model scores each caption against the facts.

Catches what the rule-based validator can't, like invented scenes ("in the attic").
The judge sees text only - the page facts and the scene labels detected in the photos.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Final

from ..ollama import OllamaError, chat
from ..photo import Photo
from ..pipeline import PageResult

DEFAULT_JUDGE_MODEL: Final = "llama3.1:8b"

GENERIC_LABELS: Final = frozenset(
    {
        "Person",
        "Human",
        "People",
        "Apparel",
        "Clothing",
        "Face",
        "Photo",
        "Photography",
        "Portrait",
        "Female",
        "Male",
        "Girl",
        "Boy",
        "Woman",
        "Man",
    }
)

RUBRIC: Final = """You review captions for a printed photo book.
You are given what is known about the photos on one page - date, place, occasion and
the things a vision model detected in them - and the caption written for the page.

Score the caption from 1 to 5 on each of:
- faithful: 5 = every claim is supported by what is known, 1 = mostly invented.
  Rooms, activities, weather or people that are not in the known facts count as invented.
- specific: 5 = says what the page is about, 1 = could be any page.
- natural: 5 = reads like a person titled their own photos, 1 = awkward or robotic.

List any words or phrases not supported by what is known in "unsupported".
Reply with JSON only:
{"faithful": n, "specific": n, "natural": n, "unsupported": ["..."]}"""


@dataclass(frozen=True, slots=True)
class Verdict:
    faithful: int
    specific: int
    natural: int
    unsupported: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scene_labels(
    photos: Sequence[Photo], min_confidence: float = 0.7, limit: int = 12
) -> list[str]:
    """Most confident non-generic labels across the page."""
    best: dict[str, float] = {}
    for p in photos:
        for name, conf in p.confidences(min_confidence).items():
            if name not in GENERIC_LABELS:
                best[name] = max(conf, best.get(name, 0.0))
    return sorted(best, key=lambda n: -best[n])[:limit]


def known_facts(result: PageResult, photos: Sequence[Photo]) -> dict[str, Any]:
    f = result.facts
    return {
        "date": f.date_text,
        "place": f.place,
        "occasion": f.occasion,
        "season": f.season,
        "setting": f.setting,
        "kids": f.kids,
        "family": f.family,
        "detected_in_photos": scene_labels(photos),
    }


def _score(value: Any) -> int:
    return min(5, max(1, int(value)))


class Judge:
    """Scores captions with a local model through Ollama."""

    def __init__(
        self, model: str = DEFAULT_JUDGE_MODEL, host: str | None = None, timeout: float = 60.0
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        messages = [{"role": "system", "content": RUBRIC}, {"role": "user", "content": prompt}]
        return chat(messages, self.model, self.host, self.timeout, temperature=0.0, json_mode=True)

    def judge(self, caption: str, facts: dict[str, Any]) -> Verdict | None:
        """Verdict for one caption, or None if the judge failed or replied badly."""
        prompt = json.dumps({"known": facts, "caption": caption})
        try:
            reply = json.loads(self.complete(prompt))
            return Verdict(
                faithful=_score(reply["faithful"]),
                specific=_score(reply["specific"]),
                natural=_score(reply["natural"]),
                unsupported=[str(u) for u in reply.get("unsupported", [])],
            )
        except (OllamaError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
