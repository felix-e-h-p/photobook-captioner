from datetime import datetime

import pytest

from captioner.dates import format_dates, ordinal, season, shorten_months, strip_date


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (31, "31st"),
    ],
)
def test_ordinal(n: int, expected: str) -> None:
    assert ordinal(n) == expected


def test_format_dates_single_day() -> None:
    assert format_dates([datetime(2019, 2, 7, 10)], "day") == "7th February 2019"
    assert format_dates([datetime(2019, 2, 7, 10)], "month") == "February 2019"


def test_format_dates_ranges() -> None:
    two_days = [datetime(2018, 9, 13), datetime(2018, 9, 14)]
    assert format_dates(two_days, "day") == "13th–14th September 2018"
    assert format_dates(two_days, "month") == "September 2018"
    assert format_dates([datetime(2019, 1, 30), datetime(2019, 2, 2)], "day") == "Jan–Feb 2019"
    assert format_dates([datetime(2019, 12, 30), datetime(2020, 1, 2)], "day") == "2019–2020"


def test_format_dates_empty() -> None:
    assert format_dates([], "day") is None


@pytest.mark.parametrize(
    ("month", "north", "south"),
    [
        (12, "winter", "summer"),
        (1, "winter", "summer"),
        (3, "spring", "autumn"),
        (6, "summer", "winter"),
        (8, "summer", "winter"),
        (9, "autumn", "spring"),
        (11, "autumn", "spring"),
    ],
)
def test_season_flips_south_of_the_equator(month: int, north: str, south: str) -> None:
    when = datetime(2018, month, 8)
    assert season(when, 51.5) == north
    assert season(when, -33.9) == south
    assert season(when, None) == north


def test_shorten_months() -> None:
    assert shorten_months("13th–14th September 2018") == "13th–14th Sep 2018"
    assert shorten_months("Jan–Feb 2019") == "Jan–Feb 2019"


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("14th September 2018 – Baking treats at home", "Baking treats at home"),
        ("Ski week in Yuzawa, Japan, Feb 2020", "Ski week in Yuzawa Japan"),
        ("13th–14th Sept 2018 – Baking", "Baking"),
    ],
)
def test_strip_date(caption: str, expected: str) -> None:
    assert strip_date(caption) == expected
