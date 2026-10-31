"""Run an evaluation - sample pages from each folder, caption them, score them."""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..photo import load_photo, photos_in
from ..pipeline import PageResult, caption_page
from ..writers import Writer
from .clip import ImageTextScorer
from .judge import Judge, known_facts
from .metrics import PageRecord, page_record
from .sampling import Mode, dedupe, sample_pages

log = logging.getLogger(__name__)


def _judge(judge: Judge, result: PageResult) -> dict[str, Any] | None:
    facts = known_facts(result, [load_photo(p) for p in result.images])
    verdict = judge.judge(result.caption, facts)
    return verdict.to_dict() if verdict else None


def _clip(clip: ImageTextScorer, result: PageResult) -> tuple[float, float]:
    """Similarity of the caption, and of the template caption as a baseline."""
    return (
        clip.similarity(result.images, result.caption),
        clip.similarity(result.images, result.template_caption),
    )


def evaluate(
    data_dir: str | Path,
    expected: Mapping[str, str | None],
    writer: Writer,
    use_context: bool = True,
    mode: Mode = "random",
    sizes: Sequence[int] = (1, 2, 3),
    pages_per_size: int = 30,
    remove_duplicates: bool = True,
    seed: int = 0,
    home_country: str = "GB",
    judge: Judge | None = None,
    clip: ImageTextScorer | None = None,
) -> list[PageRecord]:
    """Caption sampled pages from every folder in `expected` and score each one.

    The whole folder is the album, so with `use_context` each page can draw on photos
    from the same session, as it would in a real book. `judge` and `clip` add the
    optional model-based scores.
    """
    records: list[PageRecord] = []
    for i, (folder, occasion) in enumerate(sorted(expected.items())):
        album = photos_in(Path(data_dir) / folder)
        pool = dedupe(album) if remove_duplicates else album
        pages = sample_pages(pool, sizes, pages_per_size, mode, seed + i)
        for page in pages:
            result = caption_page(page, album if use_context else None, writer, home_country)
            clip_scores = _clip(clip, result) if clip else (None, None)
            records.append(
                page_record(
                    result,
                    folder,
                    occasion,
                    judge=_judge(judge, result) if judge else None,
                    clip=clip_scores[0],
                    clip_template=clip_scores[1],
                )
            )
        log.info("%s - %d pages captioned", folder, len(pages))
    return records
