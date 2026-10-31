"""Dates in captions - written out from capture times, and found again in caption text.

One place for month names and the year pattern, so the writer, the validator and the
evaluation all agree on what a date looks like.
"""

import re
from collections.abc import Sequence
from datetime import datetime
from typing import Final

from .config import DateStyle

MONTHS: Final = (
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
)
MONTH_ABBREVIATIONS: Final = (*(m[:3] for m in MONTHS), "Sept")
MONTH_WORDS: Final = frozenset(m.lower() for m in MONTHS + MONTH_ABBREVIATIONS)

YEAR: Final = re.compile(r"\b(?:19|20)\d{2}\b")
_LONG_MONTH: Final = re.compile(rf"\b({'|'.join(m[:3] for m in MONTHS)})[a-z]+\b")
_DATE_BITS: Final = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\b|{YEAR.pattern}|"
    rf"\b(?:{'|'.join(m[:3] for m in MONTHS)})[a-z]*\b|[–,]"
)

_SEASONS: Final = ("winter", "spring", "summer", "autumn")


def season(when: datetime, lat: float | None) -> str:
    """Meteorological season, flipped south of the equator.

    Dec-Feb winter, Mar-May spring and so on. South of the equator, two seasons on.
    """
    shift = 2 if lat is not None and lat < 0 else 0
    return _SEASONS[(when.month % 12 // 3 + shift) % 4]


def ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def format_dates(dates: Sequence[datetime], style: DateStyle) -> str | None:
    """'day' - 7th February 2019, 13th–14th September 2018.
    'month' - February 2019. Longer spans - Jan–Feb 2019, 2019–2020."""
    if not dates:
        return None
    lo, hi = min(dates), max(dates)
    if lo.year != hi.year:
        return f"{lo.year}–{hi.year}"
    if lo.month != hi.month:
        return f"{lo:%b}–{hi:%b %Y}"
    if style == "day":
        if lo.day == hi.day:
            return f"{ordinal(lo.day)} {lo:%B %Y}"
        return f"{ordinal(lo.day)}–{ordinal(hi.day)} {lo:%B %Y}"
    return f"{lo:%B %Y}"


def shorten_months(text: str) -> str:
    """'September 2018' -> 'Sep 2018'."""
    return _LONG_MONTH.sub(r"\1", text)


def strip_date(caption: str) -> str:
    """'14th September 2018 – Baking treats' -> 'Baking treats'."""
    return " ".join(_DATE_BITS.sub(" ", caption).split())
