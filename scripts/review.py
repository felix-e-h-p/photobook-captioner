"""Human review - rate a sample of captions by hand, then log the scores to MLflow.

    uv run python scripts/review.py make <run_id>
    (open outputs/review/<run_id>/review.html to see the photos, fill in review.csv)
    uv run python scripts/review.py score <run_id>

Ratings are 1-5. faithful - is everything in the caption true of the photos?
natural - would you be happy to see it printed in your own book?
"""

import argparse
import csv
import html
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
OUT_DIR = Path("outputs/review")
FIELDS = ["id", "folder", "images", "caption", "template_caption", "faithful", "natural", "notes"]


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


def _html(rows: list[dict[str, str]], out: Path) -> str:
    cards = []
    for row in rows:
        images = "".join(
            f'<img src="{html.escape(os.path.relpath(p, out))}">' for p in row["images"].split(";")
        )
        cards.append(
            f"<section><h3>{row['id']}. {html.escape(row['caption'])}</h3>"
            f"<p>template: {html.escape(row['template_caption'])}</p>{images}</section>"
        )
    style = "img{height:180px;margin:4px}section{margin-bottom:24px}body{font-family:sans-serif}"
    return f"<html><head><style>{style}</style></head><body>{''.join(cards)}</body></html>"


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
    (out / "review.html").write_text(_html(rows, out))
    print(f"{len(rows)} pages to review - fill in {out / 'review.csv'}, photos in review.html")


def score(run_id: str) -> None:
    sheet = OUT_DIR / run_id / "review.csv"
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
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    if args.command == "make":
        make(args.run_id, args.n, args.seed)
    else:
        score(args.run_id)


if __name__ == "__main__":
    main()
