"""Capture time and GPS from EXIF, scene labels from the file next to it."""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES: Final = frozenset({".jpg", ".jpeg", ".png", ".heic"})

# EXIF tag ids
_EXIF_IFD: Final = 0x8769
_GPS_IFD: Final = 0x8825
_DATETIME: Final = 306
_DATETIME_ORIGINAL: Final = 36867
_GPS_LAT_REF, _GPS_LAT, _GPS_LON_REF, _GPS_LON = 1, 2, 3, 4

# AWS CLI output quotes values but not keys.
# Quoting a key that follows {, [ or , gives valid JSON.
# Label names are consistently quoted.
_KEY_RE: Final = re.compile(r"([{,\[]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:")


@dataclass(frozen=True, slots=True)
class Label:
    name: str
    confidence: float
    parents: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Photo:
    path: str
    taken_at: datetime | None = None
    lat: float | None = None
    lon: float | None = None
    labels: tuple[Label, ...] = ()

    @property
    def has_gps(self) -> bool:
        return self.lat is not None and self.lon is not None

    def confidences(self, min_confidence: float = 0.0) -> dict[str, float]:
        """Label name -> confidence, for labels at or above `min_confidence`."""
        return {lab.name: lab.confidence for lab in self.labels if lab.confidence >= min_confidence}


def parse_label_file(path: str | Path) -> tuple[Label, ...]:
    data = json.loads(_KEY_RE.sub(r'\1"\2":', Path(path).read_text()))
    return tuple(
        Label(
            name=lab["Name"],
            confidence=float(lab["Confidence"]) / 100,
            parents=tuple(p["Name"] for p in lab.get("Parents", [])),
        )
        for lab in data.get("Labels", [])
    )


def _to_degrees(dms: Any, ref: str) -> float:
    deg = float(dms[0]) + float(dms[1]) / 60 + float(dms[2]) / 3600
    return -deg if ref in ("S", "W") else deg


def _parse_datetime(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def read_exif(path: str | Path) -> tuple[datetime | None, float | None, float | None]:
    """Capture time, latitude, longitude. Any can be None."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
    except (OSError, UnidentifiedImageError):
        return None, None, None

    taken_at = _parse_datetime(exif.get_ifd(_EXIF_IFD).get(_DATETIME_ORIGINAL))
    taken_at = taken_at or _parse_datetime(exif.get(_DATETIME))

    lat = lon = None
    gps = exif.get_ifd(_GPS_IFD)
    try:
        if _GPS_LAT in gps and _GPS_LON in gps:
            lat = _to_degrees(gps[_GPS_LAT], gps.get(_GPS_LAT_REF, "N"))
            lon = _to_degrees(gps[_GPS_LON], gps.get(_GPS_LON_REF, "E"))
            if lat == 0 and lon == 0:
                lat = lon = None
    except (TypeError, ValueError, ZeroDivisionError, IndexError):
        lat = lon = None
    return taken_at, lat, lon


@lru_cache(maxsize=4096)
def load_photo(path: str) -> Photo:
    """EXIF plus labels for one photo. Missing label file means no labels, not an error."""
    taken_at, lat, lon = read_exif(path)
    label_path = Path(f"{path}.json")
    labels = parse_label_file(label_path) if label_path.exists() else ()
    return Photo(path=str(path), taken_at=taken_at, lat=lat, lon=lon, labels=labels)


def by_time(paths: Iterable[str]) -> list[str]:
    """Photos in the order they were taken, name order if no date."""
    return sorted(paths, key=lambda p: (load_photo(p).taken_at or datetime.max, Path(p).name))


def photos_in(folder: str | Path) -> list[str]:
    """Photos in a folder, in the order they were taken."""
    return by_time(str(p) for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
