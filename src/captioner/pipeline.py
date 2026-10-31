"""End to end: image paths in, caption out."""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from os import PathLike
from typing import Any, Final

from .config import Config, load_config
from .facts import PageFacts, build_facts
from .photo import Photo, load_photo
from .writers import TemplateWriter, Writer, validate

type StrPath = str | PathLike[str]

SESSION_RADIUS_KM: Final = 25


@dataclass(frozen=True, slots=True)
class PageResult:
    """Caption plus everything that went into it. Used by evaluation and for logging.

    `rejected` lists the validator's problems when a writer's caption was thrown out
    in favour of the template.
    """

    images: list[str]
    caption: str
    template_caption: str
    writer: str
    facts: PageFacts
    n_context: int = 0
    rejected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _km(a: Photo, b: Photo) -> float:
    if a.lat is None or a.lon is None or b.lat is None or b.lon is None:
        return math.inf
    dlat, dlon = math.radians(b.lat - a.lat), math.radians(b.lon - a.lon)
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(a.lat)) * math.cos(math.radians(b.lat)) * math.sin(dlon / 2) ** 2
    )
    return 6371 * 2 * math.asin(math.sqrt(h))


def same_session(
    page: Sequence[Photo], candidates: Sequence[Photo], window_hours: float
) -> list[Photo]:
    """Other photos taken close in time and place to the page."""
    window = timedelta(hours=window_hours)
    times = [p.taken_at for p in page if p.taken_at]
    if not times:
        return []
    page_paths = {p.path for p in page}
    page_has_gps = any(p.has_gps for p in page)
    out = []
    for c in candidates:
        if c.path in page_paths or c.taken_at is None:
            continue
        if not any(abs(c.taken_at - t) <= window for t in times):
            continue
        if page_has_gps and not any(_km(c, p) <= SESSION_RADIUS_KM for p in page):
            continue
        out.append(c)
    return out


def caption_page(
    image_paths: Sequence[StrPath],
    album_paths: Sequence[StrPath] | None = None,
    writer: Writer | None = None,
    home_country: str = "GB",
    config: Config | None = None,
) -> PageResult:
    """Caption for a page, with the facts behind it."""
    if not 1 <= len(image_paths) <= 3:
        raise ValueError(f"a page holds 1 to 3 photos, got {len(image_paths)}")
    config = config or load_config()
    writer = writer or TemplateWriter()

    photos = [load_photo(str(p)) for p in image_paths]
    context = None
    if album_paths:
        album = [load_photo(str(p)) for p in album_paths]
        context = same_session(photos, album, config.settings.context_window_hours) or None

    facts = build_facts(photos, context, home_country=home_country, config=config)
    template = TemplateWriter().write(facts)
    caption = writer.write(facts)
    rejected = validate(caption, facts) if writer.name != "template" else []
    if rejected:
        caption = template

    return PageResult(
        images=[str(p) for p in image_paths],
        caption=caption,
        template_caption=template,
        writer=writer.name,
        facts=facts,
        n_context=len(context or []),
        rejected=rejected,
    )


def generate_caption(
    image_paths: list[str],
    album_paths: list[str] | None = None,
    writer: Writer | None = None,
    home_country: str = "GB",
) -> str:
    """Caption for a photobook page of 1 to 3 photos.

    Args:
        image_paths: photos on the page. Scene labels are read from "<photo>.json"
            next to each photo, if present.
        album_paths: optional, other photos in the book. Only used to find photos
            from the same session as extra scene evidence.
        writer: TemplateWriter() by default, or another writer such as LLMWriter.
        home_country: ISO country code of the user. Places abroad get the country
            added ("Yuzawa, Japan").
    """
    return caption_page(image_paths, album_paths, writer, home_country).caption
