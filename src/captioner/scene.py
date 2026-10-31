"""Occasion on a page determined from scene labels.

Labels mapped onto the occasions in occasions.toml rather than used as caption text.

For each occasion:

- each photo scores sum(weight x confidence) over the occasion's evidence labels
- page score is the average of mean and max over photos
- only eligible if a core label is on the page and no blocked_by label is
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from .config import Config, Occasion, load_config
from .photo import Photo

type Source = Literal["page", "context"]
type PageSetting = Literal["indoor", "outdoor", "unknown"]
type Pool = Literal["page", "context"]

_INDOOR: Final = {"Indoors": 1.0, "Room": 0.7, "Living Room": 1.0, "Kitchen": 1.0, "Bedroom": 1.0}
_OUTDOOR: Final = {
    "Outdoors": 1.0,
    "Nature": 0.7,
    "Garden": 1.0,
    "Yard": 1.0,
    "Beach": 1.0,
    "Snow": 1.0,
}
_KIDS: Final = {"Child": 1.0, "Kid": 1.0, "Baby": 1.0}
_FAMILY: Final = {"Family": 1.0}


@dataclass(frozen=True, slots=True)
class SceneResult:
    occasion: Occasion | None
    source: Source | None
    score: float
    runner_up: str | None
    margin: float
    evidence: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    setting: PageSetting = "unknown"
    kids: bool = False
    family: bool = False


def _photo_score(photo: Photo, occasion: Occasion, min_conf: float) -> float:
    conf = photo.confidences(min_conf)
    return sum(w * conf.get(name, 0.0) for name, w in occasion.evidence.items())


def _labels_on(photos: Sequence[Photo], min_conf: float) -> set[str]:
    return {name for p in photos for name in p.confidences(min_conf)}


def _mean_signal(photos: Sequence[Photo], weights: Mapping[str, float], min_conf: float) -> float:
    if not photos:
        return 0.0
    total = 0.0
    for p in photos:
        conf = p.confidences(min_conf)
        total += sum(w * conf.get(name, 0.0) for name, w in weights.items())
    return total / len(photos)


def page_setting(photos: Sequence[Photo], min_conf: float = 0.5) -> PageSetting:
    indoor = _mean_signal(photos, _INDOOR, min_conf)
    outdoor = _mean_signal(photos, _OUTDOOR, min_conf)
    if indoor > outdoor + 0.2:
        return "indoor"
    if outdoor > indoor + 0.2:
        return "outdoor"
    return "unknown"


def score_occasions(
    photos: Sequence[Photo], config: Config, pool: Pool = "page"
) -> dict[str, float]:
    """Score of every occasion. Ineligible ones score 0.

    pool="page" - a page of 1-3 photos. Average of mean and max over photos. A
        blocked_by label anywhere on the page rules the occasion out.

    pool="context" - many photos from one session. Plain mean, and blocked_by only
        zeroes the photo it's on, so one odd label can't veto a whole session.
    """
    if not photos:
        return {o.name: 0.0 for o in config.occasions}
    min_conf = config.settings.min_label_confidence
    present = _labels_on(photos, min_conf)
    out: dict[str, float] = {}
    for occ in config.occasions:
        if not (occ.core & present) or (pool == "page" and occ.blocked_by & present):
            out[occ.name] = 0.0
            continue
        per_photo = [
            0.0
            if occ.blocked_by & p.confidences(min_conf).keys()
            else _photo_score(p, occ, min_conf)
            for p in photos
        ]
        mean = sum(per_photo) / len(per_photo)
        out[occ.name] = mean if pool == "context" else (mean + max(per_photo)) / 2
    return out


def _ranked(scores: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _can_borrow(
    occ: Occasion, page: Sequence[Photo], setting: PageSetting, min_conf: float
) -> bool:
    """Page can take session's occasion only if nothing on it argues against it."""
    if occ.blocked_by & _labels_on(page, min_conf):
        return False
    return occ.setting == "any" or setting in ("unknown", occ.setting)


def understand_scene(
    photos: Sequence[Photo],
    context: Sequence[Photo] | None = None,
    config: Config | None = None,
) -> SceneResult:
    """Occasion for page of photos, optionally helped by identical session photos."""
    config = config or load_config()
    s = config.settings
    min_conf = s.min_label_confidence

    scores = score_occasions(photos, config)
    (best, best_score), (second_name, second_score) = _ranked(scores)[:2]
    second: str | None = second_name
    setting = page_setting(photos, min_conf)

    # What does the session as a whole clearly show, if anything?
    session: Occasion | None = None
    session_score = session_second = 0.0
    session_runner_up: str | None = None
    if context:
        ctx = _ranked(score_occasions(context, config, pool="context"))
        (c_best, session_score), (session_runner_up, session_second) = ctx[:2]
        clear = session_score - session_second >= s.context_margin
        if session_score >= s.context_threshold and clear:
            session = config.occasion(c_best)

    page_ok = best_score >= s.page_threshold
    page_wins = page_ok and (
        session is None or session.name == best or best_score >= s.page_override_threshold
    )

    occasion: Occasion | None = None
    source: Source | None = None
    if page_wins:
        occasion, source = config.occasion(best), "page"
    elif session is not None and _can_borrow(session, photos, setting, min_conf):
        occasion, source = session, "context"
        best_score, second, second_score = session_score, session_runner_up, session_second
    elif page_ok:
        occasion, source = config.occasion(best), "page"

    evidence: list[str] = []
    if occasion is not None:
        pool = photos if source == "page" else context or []
        evidence = sorted(_labels_on(pool, min_conf) & occasion.evidence.keys())

    everyone = [*photos, *(context or [])]
    return SceneResult(
        occasion=occasion,
        source=source,
        score=round(best_score, 3),
        runner_up=second if second_score > 0 else None,
        margin=round(best_score - second_score, 3),
        evidence=evidence,
        scores={k: round(v, 3) for k, v in scores.items() if v > 0},
        setting=setting,
        kids=_mean_signal(photos, _KIDS, min_conf) >= 0.6,
        family=_mean_signal(everyone, _FAMILY, min_conf) >= 0.5,
    )
