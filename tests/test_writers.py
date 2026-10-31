import pytest

from captioner.writers import TemplateWriter, validate

from .conftest import page_facts

WRITER = TemplateWriter()


def test_trip_shape() -> None:
    assert WRITER.write(page_facts()) == "Skiing holiday in Yuzawa, Japan, February 2020"


def test_one_off_occasion_shape() -> None:
    f = page_facts(
        occasion="Birthday party",
        phrase="Birthday party",
        date_text="7th February 2019",
        date_style="day",
        use_place=False,
    )
    assert WRITER.write(f) == "7th February 2019 – Birthday party"


def test_no_occasion_falls_back_to_place_and_date() -> None:
    f = page_facts(occasion=None, phrase=None, place="Knutsford", date_text="September 2018")
    assert WRITER.write(f) == "Knutsford, September 2018"


def test_place_that_echoes_the_occasion() -> None:
    f = page_facts(occasion="Beach day", phrase="Beach day", place="Bondi Beach, Australia")
    assert WRITER.write(f) == "Day at Bondi Beach, Australia, February 2020"


def test_nothing_known() -> None:
    f = page_facts(occasion=None, phrase=None, place=None, place_detail=None, date_text=None)
    assert WRITER.write(f) == "Memories"


def test_template_output_always_validates() -> None:
    f = page_facts()
    assert validate(WRITER.write(f), f) == []


@pytest.mark.parametrize(
    ("caption", "problem"),
    [
        ("Skiing holiday in Yuzawa, February 2021", "year 2021"),
        ("Skiing holiday in Yuzawa, March 2020", "month March"),
        ("Skiing with Grandma in Yuzawa, February 2020", "'Grandma'"),
        ("Skiing holiday in Niseko, February 2020", "'Niseko'"),
        ("12th February 2020 – Skiing", "day 12"),
        ("Skiing solo in Yuzawa, Jan 2020", "'solo'"),
        ("Skiing with the kids in Yuzawa, Feb 2020", "kids"),
        ("Family ski trip, Feb 2020", "family"),
        ("Autumn beach day in Yuzawa", "year missing"),
        ("", "empty"),
        ("A very long caption " * 5, "too long"),
    ],
)
def test_validate_catches_invented_facts(caption: str, problem: str) -> None:
    assert any(problem in p for p in validate(caption, page_facts()))


def test_validate_allows_rewording() -> None:
    assert validate("Ski week in Yuzawa, Feb 2020", page_facts()) == []


def test_validate_allows_sept() -> None:
    f = page_facts(date_text="14th September 2018", date_style="day")
    assert validate("Baking, 14th Sept 2018", f) == []


def test_validate_allows_kids_and_family_when_labels_show_them() -> None:
    f = page_facts(kids=True, family=True)
    assert validate("Family ski trip with the kids, Feb 2020", f) == []
