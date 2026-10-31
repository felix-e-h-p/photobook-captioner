"""Human review - rate a sample of captions by hand, then log the scores to MLflow.

    uv run python scripts/review.py make <run_id>
    (rate each page in the review page that opens, then click "Download ratings")
    uv run python scripts/review.py score <run_id>

Ratings are 1-5. faithful - is everything in the caption true of the photos?
natural - would you be happy to see it printed in your own book?
"""

import argparse
import csv
import json
import os
import random
import shutil
import webbrowser
from collections import defaultdict
from pathlib import Path
from statistics import mean
from string import Template
from typing import Any, Final

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
OUT_DIR = Path("outputs/review")
FIELDS = ["id", "folder", "images", "caption", "template_caption", "faithful", "natural", "notes"]
PAGE: Final = Template(Path(__file__).with_name("review_page.html").read_text())


def _pages(run_id: str) -> list[dict[str, Any]]:
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="pages.jsonl")
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _sample(pages: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Same number of pages from each folder, no repeated captions where possible."""
    by_folder: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        by_folder[page["folder"]].append(page)
    rng = random.Random(seed)
    per_folder = max(1, n // len(by_folder))
    picked = []
    for group in by_folder.values():
        rng.shuffle(group)
        unique = list({p["caption"]: p for p in group}.values())
        picked += (unique + [p for p in group if p not in unique])[:per_folder]
    return picked


def _html(rows: list[dict[str, str]], out: Path, run_id: str) -> str:
    """Review page - photos, both captions, and 1-5 buttons. Ratings are kept in the
    browser between visits and downloaded as a CSV at the end."""
    pages = [
        row | {"src": {p: os.path.relpath(p, out) for p in row["images"].split(";")}}
        for row in rows
    ]
    return PAGE.substitute(rows=json.dumps(pages), run_id=run_id, fields=json.dumps(FIELDS))


def make(run_id: str, n: int, seed: int) -> None:
    out = OUT_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": str(i),
            "folder": page["folder"],
            "images": ";".join(page["images"]),
            "caption": page["caption"],
            "template_caption": page["template_caption"],
            "faithful": "",
            "natural": "",
            "notes": "",
        }
        for i, page in enumerate(_sample(_pages(run_id), n, seed), start=1)
    ]
    with (out / "review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    page = out / "review.html"
    page.write_text(_html(rows, out, run_id))
    webbrowser.open(page.resolve().as_uri())
    print(f"{len(rows)} pages to review in {page} - click Download ratings when done")


def score(run_id: str, ratings: Path | None) -> None:
    """Ratings from the review page download, kept next to the page and logged to the run."""
    ratings = ratings or Path.home() / "Downloads" / f"review-{run_id}.csv"
    if not ratings.exists():
        raise SystemExit(f"{ratings} not found - click Download ratings on the review page first")
    sheet = OUT_DIR / run_id / "review.csv"
    if ratings.resolve() != sheet.resolve():
        shutil.copy(ratings, sheet)
    with sheet.open() as f:
        rows = [r for r in csv.DictReader(f) if r["faithful"] and r["natural"]]
    if not rows:
        raise SystemExit(f"no ratings in {sheet} yet")
    faithful = [int(r["faithful"]) for r in rows]
    natural = [int(r["natural"]) for r in rows]
    metrics = {
        "review/pages": float(len(rows)),
        "review/faithful": round(mean(faithful), 3),
        "review/natural": round(mean(natural), 3),
        "review/faithful_4plus": round(sum(s >= 4 for s in faithful) / len(rows), 3),
        "review/natural_4plus": round(sum(s >= 4 for s in natural) / len(rows), 3),
    }
    client = MlflowClient()
    for key, value in metrics.items():
        client.log_metric(run_id, key, value)
    client.log_artifact(run_id, str(sheet))
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    make_cmd = sub.add_parser("make", help="sample pages into a review sheet")
    make_cmd.add_argument("run_id")
    make_cmd.add_argument("--n", type=int, default=32)
    make_cmd.add_argument("--seed", type=int, default=0)
    score_cmd = sub.add_parser("score", help="log the filled-in ratings to the run")
    score_cmd.add_argument("run_id")
    score_cmd.add_argument(
        "--file", type=Path, help="ratings CSV (default ~/Downloads/review-<run_id>.csv)"
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    if args.command == "make":
        make(args.run_id, args.n, args.seed)
    else:
        score(args.run_id, args.file)


if __name__ == "__main__":
    main()
