"""Page facts to caption and acheck of captions against the facts."""

import re
from typing import Final, Protocol

from .dates import MONTH_ABBREVIATIONS, MONTH_WORDS, MONTHS, YEAR
from .facts import PageFacts

MAX_LENGTH: Final = 60

_WORD: Final = re.compile(r"[A-Za-zÀ-ÿ']+")
_ALWAYS_ALLOWED: Final = {"I", "BBQ", "Day", "Family", "The"}
_PEOPLE_WORDS: Final = {
    "solo",
    "alone",
    "friend",
    "friends",
    "couple",
    "mum",
    "mom",
    "dad",
    "grandma",
    "grandpa",
    "granny",
    "grandad",
    "grandkids",
    "grandchildren",
    "husband",
    "wife",
    "boyfriend",
    "girlfriend",
    "son",
    "daughter",
    "brother",
    "sister",
    "baby",
}
_KIDS_WORDS: Final = {"kids", "kid", "children", "child"}
_SCENE_WORDS: Final = {
    "spring",
    "summer",
    "autumn",
    "winter",
    "dark",
    "night",
    "evening",
    "morning",
    "sunset",
    "sunrise",
    "sunny",
    "sunshine",
    "snow",
    "snowy",
    "rain",
    "rainy",
    "storm",
    "misty",
    "foggy",
    "kitchen",
    "attic",
    "bedroom",
    "lounge",
    "sofa",
    "indoors",
    "inside",
    "garden",
    "outdoors",
    "outside",
    "park",
    "woods",
    "forest",
    "field",
    "countryside",
    "mountains",
    "lake",
    "river",
    "sea",
    "beach",
}
_SETTING_WORDS: Final = {
    "indoor": {"indoors", "inside"},
    "outdoor": {"outdoors", "outside", "garden"},
}


class WriterError(Exception):
    """A writer couldn't produce a caption - network, auth, timeout, empty reply."""


class Writer(Protocol):
    name: str

    def write(self, facts: PageFacts) -> str: ...


class TemplateWriter:
    """Two shapes, following the examples in the brief:

    - days out and trips - "Skiing holiday in Yuzawa, Japan, February 2020"
    - one-off occasions - "7th February 2019 - Birthday party"

    No occasion falls back to place and date - "Knutsford, September 2018". If the
    place repeats the occasion, it reads "Day at Bondi Beach" rather than "Beach day in
    Bondi Beach".
    """

    name = "template"

    def write(self, facts: PageFacts) -> str:
        f = facts
        place = f.place if f.use_place else None
        body: str | None
        if f.phrase and place and f.place_echoes_phrase:
            body = f"{'Family day' if f.family else 'Day'} at {place}"
        elif f.phrase and place:
            body = f"{f.phrase} in {place}"
        else:
            body = f.phrase or place

        if f.date_style == "day" and f.date_text and f.phrase:
            return f"{f.date_text} – {body}"
        parts = [p for p in (body, f.date_text) if p]
        return ", ".join(parts) if parts else "Memories"


def _allowed_words(f: PageFacts) -> set[str]:
    detail = f.place_detail or {}
    sources = [
        f.phrase,
        f.occasion,
        f.place,
        f.date_text,
        f.season,
        detail.get("region"),
        detail.get("country"),
        *f.evidence,
    ]
    return {w.lower() for s in sources if s for w in _WORD.findall(s)}


def validate(caption: str, facts: PageFacts) -> list[str]:
    """Problems with a caption. Empty list means safe to show."""

    c = caption.strip()
    if not c:
        return ["empty caption"]
    problems: list[str] = []
    if len(c) > MAX_LENGTH:
        problems.append(f"too long ({len(c)} > {MAX_LENGTH} characters)")

    date_text = (facts.date_text or "").lower()
    date_numbers = set(re.findall(r"\d+", date_text))
    for year in YEAR.findall(c):
        if year not in date_numbers:
            problems.append(f"year {year} not in metadata")
    known_years = YEAR.findall(date_text)
    if known_years and not any(y in c for y in known_years):
        problems.append("year missing")
    for month in MONTHS + MONTH_ABBREVIATIONS:
        full = next(m for m in MONTHS if m.startswith(month)).lower()
        if (
            re.search(rf"\b{month}\b", c)
            and full not in date_text
            and month.lower() not in date_text
        ):
            problems.append(f"month {month} not in metadata")
    for day in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", c):
        if day not in date_numbers:
            problems.append(f"day {day} not in metadata")

    words = [w.lower() for w in _WORD.findall(c)]
    for word in sorted(set(words) & _PEOPLE_WORDS):
        problems.append(f"'{word}' says who is in the photos, which the facts don't")
    if set(words) & _KIDS_WORDS and not facts.kids:
        problems.append("mentions kids, but the labels don't show any")
    if "family" in words and not facts.family:
        problems.append("mentions family, but the labels don't show it")

    allowed = _allowed_words(facts) | MONTH_WORDS
    setting_words = _SETTING_WORDS.get(facts.setting, set())
    for word in sorted(set(words) & _SCENE_WORDS - allowed - setting_words):
        problems.append(f"'{word}' describes the scene, which the facts don't")
    for i, word in enumerate(_WORD.findall(c)):
        if i == 0 or not word[0].isupper() or word in _ALWAYS_ALLOWED:
            continue
        if word.lower() not in allowed:
            problems.append(f"'{word}' not supported by the facts")
    return problems
