"""Occasion list and thresholds that are loaded from occasions.toml."""

import tomllib
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal, get_args

type DateStyle = Literal["day", "month"]
type Setting = Literal["indoor", "outdoor", "any"]


@dataclass(frozen=True, slots=True)
class Occasion:
    name: str
    evidence: dict[str, float]
    core: frozenset[str]
    blocked_by: frozenset[str]
    date_style: DateStyle = "month"
    use_place: bool = True
    seasons: frozenset[str] = frozenset()
    setting: Setting = "any"


@dataclass(frozen=True, slots=True)
class Settings:
    min_label_confidence: float
    page_threshold: float
    context_threshold: float
    context_margin: float
    context_window_hours: float
    page_override_threshold: float


@dataclass(frozen=True, slots=True)
class Config:
    settings: Settings
    occasions: tuple[Occasion, ...]

    def occasion(self, name: str) -> Occasion:
        return next(o for o in self.occasions if o.name == name)


def _check(o: dict[str, Any]) -> None:
    name = o["name"]
    if o["date_style"] not in get_args(DateStyle.__value__):
        raise ValueError(f"{name}: date_style must be 'day' or 'month'")
    if o.get("setting", "any") not in get_args(Setting.__value__):
        raise ValueError(f"{name}: setting must be 'indoor', 'outdoor' or 'any'")
    if missing := set(o["core"]) - set(o["evidence"]):
        raise ValueError(f"{name}: core labels missing from evidence: {sorted(missing)}")


def _parse(raw: dict[str, Any]) -> Config:
    occasions = []
    for o in raw["occasion"]:
        _check(o)
        occasions.append(
            Occasion(
                name=o["name"],
                evidence={k: float(v) for k, v in o["evidence"].items()},
                core=frozenset(o["core"]),
                blocked_by=frozenset(o["blocked_by"]),
                date_style=o["date_style"],
                use_place=o["use_place"],
                seasons=frozenset(o["seasons"]),
                setting=o.get("setting", "any"),
            )
        )
    return Config(settings=Settings(**raw["settings"]), occasions=tuple(occasions))


@cache
def load_config(path: str | None = None) -> Config:
    """Bundled occasions.toml, or a custom one from `path`."""
    if path is None:
        text = resources.files("captioner").joinpath("occasions.toml").read_text()
    else:
        text = Path(path).read_text()
    return _parse(tomllib.loads(text))
