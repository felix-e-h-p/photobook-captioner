from pathlib import Path

import pytest

from captioner.config import load_config

VALID_SETTINGS = """
[settings]
min_label_confidence = 0.5
page_threshold = 1.0
context_threshold = 0.4
context_margin = 0.2
context_window_hours = 6
page_override_threshold = 3.0
"""


def test_bundled_config_loads() -> None:
    config = load_config()
    names = [o.name for o in config.occasions]
    assert {"Beach day", "Birthday party", "Skiing", "Baking"} <= set(names)
    assert len(names) == len(set(names))


def test_bundled_core_labels_are_all_evidence() -> None:
    for occ in load_config().occasions:
        assert occ.core <= occ.evidence.keys(), occ.name


def _write(tmp_path: Path, occasion: str) -> str:
    path = tmp_path / "occasions.toml"
    path.write_text(VALID_SETTINGS + occasion)
    return str(path)


def test_core_label_missing_from_evidence_is_rejected(tmp_path: Path) -> None:
    bad = """
[[occasion]]
name = "Bad"
date_style = "day"
use_place = true
seasons = []
core = ["Cake"]
blocked_by = []
evidence = { Balloon = 1 }
"""
    with pytest.raises(ValueError, match="core labels missing"):
        load_config(_write(tmp_path, bad))


def test_unknown_date_style_is_rejected(tmp_path: Path) -> None:
    bad = """
[[occasion]]
name = "Bad"
date_style = "week"
use_place = true
seasons = []
core = []
blocked_by = []
evidence = {}
"""
    with pytest.raises(ValueError, match="date_style"):
        load_config(_write(tmp_path, bad))
