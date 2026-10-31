from captioner.geo import place_name


def test_home_place_is_just_the_town() -> None:
    place = place_name(53.306, -2.358, home_country="GB")
    assert place is not None
    assert place.label == "Knutsford"
    assert not place.abroad


def test_place_abroad_adds_the_country() -> None:
    place = place_name(39.034, 140.546, home_country="GB")
    assert place is not None
    assert place.label == "Yuzawa, Japan"


def test_home_country_changes_the_label() -> None:
    place = place_name(39.034, 140.546, home_country="JP")
    assert place is not None
    assert place.label == "Yuzawa"


def test_no_gps_gives_no_place() -> None:
    assert place_name(None, None) is None
