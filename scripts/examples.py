"""Example captions for every folder - each folder laid out as a book, page by page.

    uv run python scripts/examples.py
    uv run python scripts/examples.py --llm

Photos in time order, a fixed number per page, whole folder as the album - close to how
a real book is laid out. Writes outputs/examples/<writer>/<folder>.md to read, and
outputs/examples/<writer>/captions.jsonl with every page and the facts behind it.
"""

import argparse
import json
from itertools import batched
from pathlib import Path

from captioner import PageResult, caption_page, make_writer
from captioner.photo import photos_in

DATA_DIR = Path("data")
OUT_DIR = Path("outputs/examples")


def _markdown(folder: str, pages: list[PageResult]) -> str:
    lines = [f"# {folder}", ""]
    for i, page in enumerate(pages, start=1):
        names = ", ".join(Path(p).name for p in page.images)
        lines.append(f"{i}. {page.caption}  ({names})")
        if page.writer != "template":
            lines.append(f"    - template: {page.template_caption}")
        if page.rejected:
            lines.append(f"    - rejected {page.writer_caption!r}: {'; '.join(page.rejected)}")
        lines.extend(f"    - note: {w}" for w in page.facts.warnings)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", action="store_true", help="caption with the LLM writer")
    parser.add_argument("--page-size", type=int, choices=[1, 2, 3], default=2)
    args = parser.parse_args()

    writer = make_writer("llm" if args.llm else "template")
    out = OUT_DIR / writer.name.split(":")[0]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "captions.jsonl").open("w") as jsonl:
        for folder in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
            album = photos_in(folder)
            pages = [caption_page(page, album, writer) for page in batched(album, args.page_size)]
            (out / f"{folder.name}.md").write_text(_markdown(folder.name, pages))
            jsonl.writelines(
                json.dumps(p.to_dict() | {"folder": folder.name}, default=str) + "\n" for p in pages
            )
            print(f"{folder.name} - {len(pages)} pages")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
