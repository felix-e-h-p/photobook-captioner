"""HTTP API - caption a page of photos.

    uv run uvicorn captioner.api:app --reload
    curl -F photos=@data/skiing-holiday/Image_1.jpg \\
         -F labels=@data/skiing-holiday/Image_1.jpg.json localhost:8000/caption

Uploads are written to a temporary folder for the request and deleted straight after.
Only text facts reach the writer, never the photos.
"""

import os
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from .llm import make_writer
from .pipeline import PageResult, caption_page

type WriterName = Literal["template", "llm"]


def _version() -> str:
    """Installed package version, or "dev" when run from source without installing."""
    try:
        return version("captioner")
    except PackageNotFoundError:
        return "dev"


app = FastAPI(title="Photobook captioner", version=_version())


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    version: str = app.version


class Caption(BaseModel):
    """Caption plus the facts behind it, so a client can show why."""

    caption: str
    writer: str
    template_caption: str
    occasion: str | None
    date: str | None
    place: str | None
    warnings: list[str]
    rejected: list[str]

    @classmethod
    def of(cls, result: PageResult) -> "Caption":
        f = result.facts
        return cls(
            caption=result.caption,
            writer=result.writer,
            template_caption=result.template_caption,
            occasion=f.occasion,
            date=f.date_text,
            place=f.place,
            warnings=f.warnings,
            rejected=result.rejected,
        )


class PathPage(BaseModel):
    """A page of photos the server can already read, e.g. a mounted photo library."""

    images: list[str] = Field(min_length=1, max_length=3)
    album: list[str] | None = None
    home_country: str = "GB"


def _photo_root() -> Path:
    """Only photos under CAPTIONER_PHOTO_ROOT (default the working folder) can be read by path."""
    return Path(os.environ.get("CAPTIONER_PHOTO_ROOT", ".")).resolve()


def _caption(
    images: list[str], album: list[str] | None, writer: WriterName, home_country: str
) -> Caption:
    try:
        result = caption_page(images, album, make_writer(writer), home_country)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return Caption.of(result)


@app.get("/health")
def health() -> Health:
    return Health()


@app.post("/caption")
def caption_upload(
    photos: Annotated[list[UploadFile], File(description="1 to 3 photos for the page")],
    labels: Annotated[
        list[UploadFile] | None, File(description="optional label files, named <photo>.json")
    ] = None,
    writer: Annotated[WriterName, Query()] = "template",
    home_country: Annotated[str, Query(min_length=2, max_length=2)] = "GB",
) -> Caption:
    """Caption uploaded photos. EXIF date and GPS are read from the files themselves."""
    if not 1 <= len(photos) <= 3:
        raise HTTPException(
            status_code=422, detail=f"a page holds 1 to 3 photos, got {len(photos)}"
        )
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, upload in enumerate([*photos, *(labels or [])]):
            name = Path(upload.filename or f"upload_{i}").name
            (Path(tmp) / name).write_bytes(upload.file.read())
            if i < len(photos):
                paths.append(str(Path(tmp) / name))
        return _caption(paths, None, writer, home_country)


@app.post("/caption/paths")
def caption_paths(page: PathPage, writer: Annotated[WriterName, Query()] = "template") -> Caption:
    """Caption photos by path on the server. `album` lets the page borrow from its session."""
    root = _photo_root()
    paths = [*page.images, *(page.album or [])]
    outside = [p for p in paths if not Path(p).resolve().is_relative_to(root)]
    if outside:
        raise HTTPException(status_code=403, detail=f"outside the photo root: {outside[0]}")
    return _caption(page.images, page.album, writer, page.home_country)
