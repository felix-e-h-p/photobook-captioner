"""Pages to evaluate on where near-duplicates removed first, then random or consecutive."""

import random
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps

from ..photo import load_photo

type Mode = Literal["random", "consecutive"]


def dhash(path: str | Path) -> int:
    """64-bit difference hash - which neighbouring pixels get brighter in a 9x8 thumbnail."""
    with Image.open(path) as im:
        small = ImageOps.exif_transpose(im).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = small.tobytes()
    bits = 0
    for row in range(8):
        for col in range(8):
            left, right = pixels[row * 9 + col], pixels[row * 9 + col + 1]
            bits = (bits << 1) | (right > left)
    return bits


def by_time(paths: Sequence[str]) -> list[str]:
    """Photos in the order they were taken, name order if no date."""
    return sorted(paths, key=lambda p: (load_photo(p).taken_at or datetime.max, Path(p).name))


def dedupe(paths: Sequence[str], max_bits: int = 6) -> list[str]:
    """One photo per burst. Keeps the first of each group of near-identical photos."""
    kept: list[str] = []
    hashes: list[int] = []
    for path in by_time(paths):
        h = dhash(path)
        if all((h ^ other).bit_count() > max_bits for other in hashes):
            kept.append(path)
            hashes.append(h)
    return kept


def sample_pages(
    paths: Sequence[str],
    sizes: Sequence[int] = (1, 2, 3),
    pages_per_size: int = 30,
    mode: Mode = "random",
    seed: int = 0,
) -> list[list[str]]:
    """Pages of 1-3 photos.

    random - any photos from the folder, like the exploration notebook.
    consecutive - photos next to each other in time, closer to a real book page.
    """
    rng = random.Random(seed)
    ordered = by_time(paths)
    pages: list[list[str]] = []
    for k in sizes:
        if k > len(ordered):
            continue
        for _ in range(pages_per_size):
            if mode == "random":
                pages.append(rng.sample(ordered, k))
            else:
                start = rng.randrange(len(ordered) - k + 1)
                pages.append(ordered[start : start + k])
    return pages
