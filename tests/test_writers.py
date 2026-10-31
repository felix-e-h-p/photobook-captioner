from typing import Any

import pytest

from captioner.facts import PageFacts
from captioner.writers import TemplateWriter, validate


def facts(**overrides: Any) -> PageFacts:
    base: dict[str, Any] = {
        "n_photos": 1,
        "occasion": "Skiing",
        "phrase": "Skiing holiday",
        "occasion_source": "page",
        "date_text": "February 2020",
        "date_style": "month",
        "place": "Yuzawa, Japan",
        "place_detail": {"town": "Yuzawa", "region": "Akita", "country": "Japan"},
        "use_place": True,
        "season": "winter",
        "setting": "outdoor",
        "kids": False,
        "family": False,
        "evidence": ["Skiing"],
        "occasion_score": 3.0,
        "occasion_margin": 3.0,
        "runner_up": None,
    }
    return PageFacts(**(base | overrides))


WRITER = TemplateWriter()


def test_trip_shape() -> None:
    assert WRITER.write(facts()) == "Skiing holiday in Yuzawa, Japan, February 2020"


def test_one_off_occasion_shape() -> None:
    f = facts(
        occasion="Birthday party",
        phrase="Birthday party",
        date_text="7th February 2019",
        date_style="day",
        use_place=False,
    )
    assert WRITER.write(f) == "7th February 2019 – Birthday party"


def test_no_occasion_falls_back_to_place_and_date() -> None:
    f = facts(occasion=None, phrase=None, place="Knutsford", date_text="September 2018")
    assert WRITER.write(f) == "Knutsford, September 2018"


def test_place_that_echoes_the_occasion() -> None:
    f = facts(occasion="Beach day", phrase="Beach day", place="Bondi Beach, Australia")
    assert WRITER.write(f) == "Day at Bondi Beach, Australia, February 2020"


def test_nothing_known() -> None:
    f = facts(occasion=None, phrase=None, place=None, place_detail=None, date_text=None)
    assert WRITER.write(f) == "Memories"


def test_template_output_always_validates() -> None:
    f = facts()
    assert validate(WRITER.write(f), f) == []


@pytest.mark.parametrize(
    ("caption", "problem"),
    [
        ("Skiing holiday in Yuzawa, February 2021", "year 2021"),
        ("Skiing holiday in Yuzawa, March 2020", "month March"),
        ("Skiing with Grandma in Yuzawa, February 2020", "'Grandma'"),
        ("Skiing holiday in Niseko, February 2020", "'Niseko'"),
        ("12th February 2020 – Skiing", "day 12"),
        ("", "empty"),
        ("A very long caption " * 5, "too long"),
    ],
)
def test_validate_catches_invented_facts(caption: str, problem: str) -> None:
    assert any(problem in p for p in validate(caption, facts()))


def test_validate_allows_rewording() -> None:
    assert validate("Ski week in Yuzawa, Feb 2020", facts()) == []
