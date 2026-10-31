"""Evaluate the captioner and log the run to MLflow.

    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py writer=llm mode=consecutive
    uv run python scripts/evaluate.py -m writer=template,llm use_context=true,false
    uv run python scripts/evaluate.py writer=llm judge.enabled=true clip.enabled=true

Each run logs its settings, every metric from captioner.evaluation.summarise, and the
captioned pages as artifacts. Compare runs with `uv run mlflow ui
--backend-store-uri sqlite:///mlflow.db`.
"""

import hashlib
import json
import subprocess
import tempfile
from collections import defaultdict
from importlib import resources
from pathlib import Path
from typing import Any, cast

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from captioner import make_writer
from captioner.evaluation import PageRecord, evaluate, summarise
from captioner.evaluation.clip import ClipScorer, ImageTextScorer
from captioner.evaluation.judge import Judge
from captioner.evaluation.metrics import rejection_reasons


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _examples(records: list[PageRecord], per_folder: int = 8) -> str:
    """A readable sample of captions per folder, for eyeballing in the MLflow UI."""
    by_folder: dict[str, list[PageRecord]] = defaultdict(list)
    for r in records:
        by_folder[r.folder].append(r)
    lines = ["# Example captions", ""]
    for folder, group in sorted(by_folder.items()):
        lines += [f"## {folder}", ""]
        for r in group[:per_folder]:
            names = ", ".join(Path(p).name for p in r.images)
            lines.append(f"- {r.caption}  ({names})")
            if r.rejected:
                lines.append(
                    f"  - writer said {r.writer_caption!r}, rejected: {'; '.join(r.rejected)}"
                )
            if r.judge and r.judge["unsupported"]:
                lines.append(f"  - judge flagged: {', '.join(r.judge['unsupported'])}")
        lines.append("")
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> float:
    expected = cast(dict[str, str | None], OmegaConf.to_container(cfg.expected))
    writer = make_writer(cfg.writer)
    judge = Judge(cfg.judge.model) if cfg.judge.enabled else None
    clip: ImageTextScorer | None = (
        ClipScorer(cfg.clip.model, cfg.clip.pretrained) if cfg.clip.enabled else None
    )
    records = evaluate(
        cfg.data_dir,
        expected,
        writer,
        use_context=cfg.use_context,
        mode=cfg.mode,
        sizes=list(cfg.page_sizes),
        pages_per_size=cfg.pages_per_size,
        remove_duplicates=cfg.remove_duplicates,
        seed=cfg.seed,
        home_country=cfg.home_country,
        judge=judge,
        clip=clip,
    )
    metrics = summarise(records)
    occasions = resources.files("captioner").joinpath("occasions.toml").read_text()

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    run_name = f"{cfg.writer}-{'context' if cfg.use_context else 'page'}-{cfg.mode}"
    with mlflow.start_run(run_name=run_name):
        params: dict[str, Any] = {
            "writer": writer.name,
            "use_context": cfg.use_context,
            "mode": cfg.mode,
            "page_sizes": list(cfg.page_sizes),
            "pages_per_size": cfg.pages_per_size,
            "remove_duplicates": cfg.remove_duplicates,
            "seed": cfg.seed,
            "home_country": cfg.home_country,
            "judge": cfg.judge.model if cfg.judge.enabled else "off",
            "clip": f"{cfg.clip.model}/{cfg.clip.pretrained}" if cfg.clip.enabled else "off",
            "occasions_sha": hashlib.sha256(occasions.encode()).hexdigest()[:12],
        }
        mlflow.log_params(params)
        mlflow.set_tag("git_commit", _git_commit())
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "pages.jsonl").write_text(
                "\n".join(json.dumps(r.to_dict()) for r in records) + "\n"
            )
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
            (out / "rejection_reasons.json").write_text(
                json.dumps(rejection_reasons(records), indent=2)
            )
            (out / "examples.md").write_text(_examples(records))
            (out / "occasions.toml").write_text(occasions)
            (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
            mlflow.log_artifacts(str(out))

    headline = [
        "all/accuracy",
        "all/wrong",
        "all/abstained",
        "all/writer_rejected",
        "all/distinct_captions",
        "all/judge_faithful",
        "all/clip_gain",
    ]
    print(run_name, {k: metrics[k] for k in headline if k in metrics})
    return metrics.get("all/accuracy", 0.0)


if __name__ == "__main__":
    main()
