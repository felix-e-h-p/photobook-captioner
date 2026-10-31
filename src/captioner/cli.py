"""Command line: caption one page, or entire folder page by page.

    captioner caption data/day-at-beach/Image_1.jpg data/day-at-beach/Image_2.jpg
    captioner caption data/day-at-beach/Image_1.jpg --album data/day-at-beach
    captioner folder data/summer-with-grandkids --page-size 2 --out outputs/grandkids.jsonl

Add --llm to command to use the local writer.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from .llm import LLMWriter
from .photo import load_photo
from .pipeline import PageResult, caption_page
from .writers import TemplateWriter, Writer

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}


def _photos_in(folder: Path) -> list[Path]:
    """Photos in a folder, in the order they were taken (name order if no date)."""
    photos = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]

    def key(p: Path) -> tuple[datetime, str]:
        return load_photo(str(p)).taken_at or datetime.max, p.name

    return sorted(photos, key=key)


def _writer(use_llm: bool) -> Writer:
    return LLMWriter() if use_llm else TemplateWriter()


def _print(result: PageResult, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), default=str))
        return
    names = ", ".join(Path(p).name for p in result.images)
    print(f"{result.caption}    [{names}]")
    for warning in result.facts.warnings:
        print(f"    note: {warning}")
    for reason in result.rejected:
        print(f"    writer caption rejected: {reason}")


def _caption(args: argparse.Namespace) -> int:
    album = [str(p) for p in _photos_in(args.album)] if args.album else None
    result = caption_page(args.photos, album, _writer(args.llm), args.home)
    _print(result, args.json)
    return 0


def _folder(args: argparse.Namespace) -> int:
    photos = [str(p) for p in _photos_in(args.folder)]
    if not photos:
        print(f"no photos in {args.folder}", file=sys.stderr)
        return 1
    writer = _writer(args.llm)
    pages = [photos[i : i + args.page_size] for i in range(0, len(photos), args.page_size)]
    out = args.out.open("w") if args.out else None
    try:
        for page in pages:
            result = caption_page(page, photos, writer, args.home)
            _print(result, args.json)
            if out:
                out.write(json.dumps(result.to_dict(), default=str) + "\n")
    finally:
        if out:
            out.close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="captioner", description=__doc__.splitlines()[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--llm", action="store_true", help="use the local LLM writer")
    common.add_argument("--home", default="GB", help="home country code (default GB)")
    common.add_argument("--json", action="store_true", help="print full results as JSON")

    sub = parser.add_subparsers(dest="command", required=True)

    caption = sub.add_parser("caption", parents=[common], help="caption one page of 1-3 photos")
    caption.add_argument("photos", nargs="+", help="photos on the page")
    caption.add_argument("--album", type=Path, help="folder of other photos from the same book")
    caption.set_defaults(run=_caption)

    folder = sub.add_parser("folder", parents=[common], help="caption a folder page by page")
    folder.add_argument("folder", type=Path)
    folder.add_argument("--page-size", type=int, choices=[1, 2, 3], default=2)
    folder.add_argument("--out", type=Path, help="also write results to a JSON lines file")
    folder.set_defaults(run=_folder)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.run(args))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
