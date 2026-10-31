"""Page facts to caption and acheck of captions against the facts."""

import re
from typing import Final, Protocol

from .facts import PageFacts

MAX_LENGTH: Final = 60

_MONTHS: Final = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_MONTH_ABBREVIATIONS: Final = [m[:3] for m in _MONTHS] + ["Sept"]
_MONTH_WORDS: Final = {m.lower() for m in _MONTHS + _MONTH_ABBREVIATIONS}
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
        detail.get("region"),
        detail.get("country"),
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
    for year in re.findall(r"\b(?:19|20)\d{2}\b", c):
        if year not in date_numbers:
            problems.append(f"year {year} not in metadata")
    known_years = re.findall(r"\b(?:19|20)\d{2}\b", date_text)
    if known_years and not any(y in c for y in known_years):
        problems.append("year missing")
    for month in _MONTHS + _MONTH_ABBREVIATIONS:
        full = next(m for m in _MONTHS if m.startswith(month)).lower()
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

    allowed = _allowed_words(facts) | _MONTH_WORDS
    for i, word in enumerate(_WORD.findall(c)):
        if i == 0 or not word[0].isupper() or word in _ALWAYS_ALLOWED:
            continue
        if word.lower() not in allowed:
            problems.append(f"'{word}' not supported by the facts")
    return problems
