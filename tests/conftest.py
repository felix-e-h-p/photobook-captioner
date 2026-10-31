"""Shared fixtures."""

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pytest
from PIL import Image

from captioner.facts import PageFacts
from captioner.photo import Label, Photo

type MakePhoto = Callable[..., Path]

KNUTSFORD: Final = {"lat": 53.306, "lon": -2.358}
YUZAWA: Final = {"lat": 39.034, "lon": 140.546}


def _dms(value: float) -> tuple[float, float, float]:
    value = abs(value)
    degrees = int(value)
    minutes = int((value - degrees) * 60)
    seconds = round((value - degrees - minutes / 60) * 3600, 2)
    return float(degrees), float(minutes), seconds


def _aws_cli_labels(labels: Mapping[str, float]) -> str:
    """Label file in the same not-quite-JSON format as the dataset - keys unquoted."""
    items = ",\n".join(
        f"    {{\n      Confidence: {conf * 100},\n      Instances: [],\n"
        f'      Name: "{name}",\n      Parents: []\n    }}'
        for name, conf in labels.items()
    )
    return f'{{\n  LabelModelVersion: "2.0",\n  Labels: [\n{items}\n  ]\n}}\n'


@pytest.fixture
def make_photo(tmp_path: Path) -> MakePhoto:
    """Write a small JPEG with optional EXIF date and GPS, plus an optional label file."""
    counter = iter(range(10_000))

    def _make(
        taken_at: datetime | None = None,
        lat: float | None = None,
        lon: float | None = None,
        labels: Mapping[str, float] | None = None,
        name: str | None = None,
    ) -> Path:
        path = tmp_path / (name or f"photo_{next(counter)}.jpg")
        exif = Image.Exif()
        if taken_at is not None:
            exif.get_ifd(0x8769)[36867] = taken_at.strftime("%Y:%m:%d %H:%M:%S")
        if lat is not None and lon is not None:
            gps = exif.get_ifd(0x8825)
            gps[1] = "N" if lat >= 0 else "S"
            gps[2] = _dms(lat)
            gps[3] = "E" if lon >= 0 else "W"
            gps[4] = _dms(lon)
        Image.new("RGB", (8, 8), "white").save(path, exif=exif)
        if labels is not None:
            Path(f"{path}.json").write_text(_aws_cli_labels(labels))
        return path

    return _make


def photo(labels: Mapping[str, float], taken_at: datetime | None = None) -> Photo:
    """In-memory photo for scene tests that don't need files."""
    return Photo(
        path=f"mem-{id(labels)}-{sorted(labels)}",
        taken_at=taken_at,
        labels=tuple(Label(name=n, confidence=c) for n, c in labels.items()),
    )


def page_facts(**overrides: Any) -> PageFacts:
    """Facts for a skiing page in Yuzawa, with any field overridden."""
    base: dict[str, Any] = {
        "n_photos": 1,
        "occasion": "Skiing",
        "phrase": "Skiing holiday",
        "occasion_source": "page",
        "date_text": "February 2020",
        "date_style": "month",
        "place": "Yuzawa, Japan",
        "place_detail": {"town": "Yuzawa", "region": "Akita", "country": "Japan"},
        "use_place": True,
        "season": "winter",
        "setting": "outdoor",
        "kids": False,
        "family": False,
        "evidence": ["Skiing"],
        "occasion_score": 3.0,
        "occasion_margin": 3.0,
        "runner_up": None,
    }
    return PageFacts(**(base | overrides))
