from datetime import datetime

import pytest

from captioner.facts import format_dates, ordinal, season


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


def test_season_flips_south_of_the_equator() -> None:
    march = datetime(2018, 3, 8)
    assert season(march, 51.5) == "spring"
    assert season(march, -33.9) == "autumn"
    assert season(march, None) == "spring"
