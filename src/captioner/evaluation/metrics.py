"""Scores for captioned pages.

Occasion - folder names are weak labels. Right, wrong, or none (no occasion claimed).
Grounding - years in the caption checked against the photos' own EXIF, independently
    of the pipeline's validator, and whether the town made it into the caption when it
    should have.
Writer - how often the LLM's caption was rejected, and why.
Variety - share of distinct captions per folder. A book that repeats one caption on
    every page reads badly even if each caption is right.
Judge (optional) - a larger local model's 1-5 scores for faithful, specific, natural.
CLIP (optional) - image-text similarity of the caption, and the gain over the template.
"""

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Literal

from ..dates import YEAR
from ..geo import place_name
from ..photo import load_photo
from ..pipeline import PageResult

type Outcome = Literal["right", "wrong", "none", "unlabelled"]


@dataclass(frozen=True, slots=True)
class PageRecord:
    folder: str
    images: list[str]
    n_photos: int
    caption: str
    template_caption: str
    writer: str
    occasion: str | None
    occasion_source: str | None
    expected: str | None
    outcome: Outcome
    years_grounded: bool
    place_in_caption: bool
    place_expected: bool
    writer_caption: str | None
    rejected: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    judge: dict[str, Any] | None = None
    clip: float | None = None
    clip_template: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _outcome(occasion: str | None, expected: str | None) -> Outcome:
    if expected is None:
        return "unlabelled"
    if occasion is None:
        return "none"
    return "right" if occasion == expected else "wrong"


def page_record(
    result: PageResult,
    folder: str,
    expected: str | None,
    judge: dict[str, Any] | None = None,
    clip: float | None = None,
    clip_template: float | None = None,
) -> PageRecord:
    photos = [load_photo(p) for p in result.images]
    exif_years = {str(p.taken_at.year) for p in photos if p.taken_at}
    caption_years = set(YEAR.findall(result.caption))
    towns = {pl.town for p in photos if (pl := place_name(p.lat, p.lon))}
    return PageRecord(
        folder=folder,
        images=result.images,
        n_photos=len(result.images),
        caption=result.caption,
        template_caption=result.template_caption,
        writer=result.writer,
        occasion=result.facts.occasion,
        occasion_source=result.facts.occasion_source,
        expected=expected,
        outcome=_outcome(result.facts.occasion, expected),
        years_grounded=caption_years <= exif_years if exif_years else not caption_years,
        place_in_caption=any(t.split()[-1] in result.caption for t in towns),
        place_expected=bool(towns) and result.facts.use_place,
        writer_caption=result.writer_caption,
        rejected=result.rejected,
        warnings=result.facts.warnings,
        judge=judge,
        clip=clip,
        clip_template=clip_template,
    )


def _share(flags: Iterable[bool]) -> float:
    values = list(flags)
    return round(sum(values) / len(values), 4) if values else 0.0


def _group_metrics(records: list[PageRecord]) -> dict[str, float]:
    labelled = [r for r in records if r.outcome != "unlabelled"]
    named = [r for r in labelled if r.outcome != "none"]
    out = {
        "pages": float(len(records)),
        "occasion_named": _share(r.occasion is not None for r in records),
        "occasion_borrowed": _share(r.occasion_source == "context" for r in records),
        "years_grounded": _share(r.years_grounded for r in records),
        "place_in_caption": _share(r.place_in_caption for r in records if r.place_expected),
        "writer_rejected": _share(bool(r.rejected) for r in records),
        "distinct_captions": round(len({r.caption for r in records}) / len(records), 4),
        "mean_length": round(mean(len(r.caption) for r in records), 2),
        "with_warnings": _share(bool(r.warnings) for r in records),
    }
    judged = [r.judge for r in records if r.judge]
    if judged:
        out |= {
            "judge_faithful": round(mean(j["faithful"] for j in judged), 3),
            "judge_specific": round(mean(j["specific"] for j in judged), 3),
            "judge_natural": round(mean(j["natural"] for j in judged), 3),
            "judge_unsupported": _share(bool(j["unsupported"]) for j in judged),
            "judge_failed": round(1 - len(judged) / len(records), 4),
        }
    clip_pairs = [(r.clip, r.clip_template) for r in records if r.clip is not None]
    pairs = [(c, t) for c, t in clip_pairs if t is not None]
    if pairs:
        out |= {
            "clip": round(mean(c for c, _ in pairs), 4),
            "clip_gain": round(mean(c - t for c, t in pairs), 4),
        }
    if labelled:
        out |= {
            "accuracy": _share(r.outcome == "right" for r in labelled),
            "wrong": _share(r.outcome == "wrong" for r in labelled),
            "abstained": _share(r.outcome == "none" for r in labelled),
            "precision": _share(r.outcome == "right" for r in named) if named else 0.0,
        }
    return out


def summarise(records: list[PageRecord]) -> dict[str, float]:
    """Flat metrics dict - overall, per folder and per page size - ready for MLflow."""
    if not records:
        return {}
    metrics = {f"all/{k}": v for k, v in _group_metrics(records).items()}
    groups: dict[str, list[PageRecord]] = defaultdict(list)
    for r in records:
        groups[r.folder].append(r)
        groups[f"{r.folder}/k{r.n_photos}"].append(r)
    for name, group in sorted(groups.items()):
        metrics |= {f"{name}/{k}": v for k, v in _group_metrics(group).items()}
    return metrics


def reason_kind(reason: str) -> str:
    """'Niseko' not supported by the facts -> <word> not supported by the facts."""
    if reason.startswith("writer failed"):
        return "writer failed"
    return re.sub(r"\d+", "N", re.sub(r"'[^']*'", "<word>", reason))


def rejection_reasons(records: Iterable[PageRecord]) -> dict[str, int]:
    """How often each kind of problem made the pipeline throw out the writer's caption."""
    counts = Counter(reason_kind(reason) for r in records for reason in r.rejected)
    return dict(counts.most_common())
