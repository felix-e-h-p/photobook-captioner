"""Everything known about a page in single checkable record.

Caption writers only see this record and not the photos.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Final

from .config import Config, DateStyle, load_config
from .geo import Place, place_name
from .photo import Photo
from .scene import PageSetting, SceneResult, Source, understand_scene

_NORTH_SEASON: Final = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}
_FLIP: Final = {"winter": "summer", "summer": "winter", "spring": "autumn", "autumn": "spring"}
MIXED_DATES_DAYS: Final = 31


def season(when: datetime, lat: float | None) -> str:
    """Meteorological season, flipped south of the equator."""
    s = _NORTH_SEASON[when.month]
    return _FLIP[s] if lat is not None and lat < 0 else s


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


@dataclass(slots=True, kw_only=True)
class PageFacts:
    n_photos: int
    occasion: str | None
    phrase: str | None
    occasion_source: Source | None
    date_text: str | None
    date_style: DateStyle
    place: str | None
    place_detail: dict[str, Any] | None
    use_place: bool
    season: str | None
    setting: PageSetting
    kids: bool
    family: bool
    evidence: list[str]
    occasion_score: float
    occasion_margin: float
    runner_up: str | None
    warnings: list[str] = field(default_factory=list)

    @property
    def place_echoes_phrase(self) -> bool:
        """True for e.g. 'Beach day' at 'Bondi Beach' - caption would repeat itself."""
        if not (self.phrase and self.place):
            return False
        phrase_words = {w.lower() for w in self.phrase.split()} - {"day", "trip", "holiday"}
        return any(w.lower().strip(",") in phrase_words for w in self.place.split())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _phrase(scene: SceneResult, place: Place | None) -> str | None:
    if scene.occasion is None:
        return None
    match scene.occasion.name:
        case "Skiing":
            return "Skiing holiday" if place and place.abroad else "Skiing trip"
        case "Baking" if scene.kids:
            return "Baking with the kids"
        case "Beach day" if scene.family:
            return "Family beach day"
        case name:
            return name


def _majority_place(photos: Sequence[Photo], home_country: str) -> Place | None:
    """Most common town across photos, so one stray GPS fix doesn't move the page."""
    places = [pl for p in photos if (pl := place_name(p.lat, p.lon, home_country))]
    if not places:
        return None
    top = Counter(p.label for p in places).most_common(1)[0][0]
    return next(p for p in places if p.label == top)


def build_facts(
    photos: Sequence[Photo],
    context: Sequence[Photo] | None = None,
    home_country: str = "GB",
    config: Config | None = None,
) -> PageFacts:
    config = config or load_config()
    warnings: list[str] = []
    scene = understand_scene(photos, context=context, config=config)
    occ = scene.occasion

    dates = [p.taken_at for p in photos if p.taken_at]
    if not dates and context and (dates := [p.taken_at for p in context if p.taken_at]):
        warnings.append("date taken from neighbouring photos")
    if not dates:
        warnings.append("no capture date")
    elif (max(dates) - min(dates)).days > MIXED_DATES_DAYS:
        warnings.append("photos on this page span more than a month")

    place = _majority_place(photos, home_country)
    if place is None and context and (place := _majority_place(context, home_country)):
        warnings.append("place taken from neighbouring photos")
    if place is None:
        warnings.append("no location")

    lat = next((p.lat for p in photos if p.has_gps), None)
    ssn = season(min(dates), lat) if dates else None
    if occ and ssn and occ.seasons and ssn not in occ.seasons:
        warnings.append(f"'{occ.name}' in {ssn} is unusual - date or location may be wrong")
    if occ and scene.source == "page" and scene.runner_up and scene.margin < 0.3:
        warnings.append(f"close call between '{occ.name}' and '{scene.runner_up}'")
    if scene.source == "context":
        warnings.append("occasion borrowed from same-session photos")

    date_style: DateStyle = occ.date_style if occ else "month"
    return PageFacts(
        n_photos=len(photos),
        occasion=occ.name if occ else None,
        phrase=_phrase(scene, place),
        occasion_source=scene.source,
        date_text=format_dates(dates, date_style),
        date_style=date_style,
        place=place.label if place else None,
        place_detail=asdict(place) if place else None,
        use_place=occ.use_place if occ else True,
        season=ssn,
        setting=scene.setting,
        kids=scene.kids,
        family=scene.family,
        evidence=scene.evidence,
        occasion_score=scene.score,
        occasion_margin=scene.margin,
        runner_up=scene.runner_up,
        warnings=warnings,
    )
