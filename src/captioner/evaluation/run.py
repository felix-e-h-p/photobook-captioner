"""Run an evaluation - sample pages from each folder, caption them, score them."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from ..pipeline import caption_page
from ..writers import Writer
from .metrics import PageRecord, page_record
from .sampling import Mode, dedupe, sample_pages

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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
) -> list[PageRecord]:
    """Caption sampled pages from every folder in `expected` and score each one.

    The whole folder is the album, so with `use_context` each page can draw on photos
    from the same session, as it would in a real book.
    """
    records: list[PageRecord] = []
    for i, (folder, occasion) in enumerate(sorted(expected.items())):
        album = sorted(
            str(p)
            for p in (Path(data_dir) / folder).iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES
        )
        pool = dedupe(album) if remove_duplicates else album
        pages = sample_pages(pool, sizes, pages_per_size, mode, seed + i)
        for page in pages:
            result = caption_page(page, album if use_context else None, writer, home_country)
            records.append(page_record(result, folder, occasion))
    return records
