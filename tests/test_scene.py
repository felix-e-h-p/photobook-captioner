from datetime import datetime

from captioner.config import load_config
from captioner.scene import page_setting, score_occasions, understand_scene

from .conftest import photo

CONFIG = load_config()


def test_clear_page_evidence_picks_the_occasion() -> None:
    page = [photo({"Party Hat": 0.9, "Birthday Party": 0.8, "Person": 0.99})]
    result = understand_scene(page, config=CONFIG)
    assert result.occasion is not None
    assert result.occasion.name == "Birthday party"
    assert result.source == "page"
    assert {"Party Hat", "Birthday Party"} <= set(result.evidence)


def test_generic_labels_give_no_occasion() -> None:
    page = [photo({"Person": 0.99, "Human": 0.99, "Clothing": 0.9})]
    assert understand_scene(page, config=CONFIG).occasion is None


def test_sea_and_ocean_alone_are_not_a_beach() -> None:
    page = [photo({"Sea": 0.7, "Ocean": 0.7, "Shop": 0.7})]
    assert score_occasions(page, CONFIG)["Beach day"] == 0.0


def test_snow_alone_is_not_skiing() -> None:
    page = [photo({"Snow": 0.95, "Person": 0.99, "Outdoors": 0.9})]
    assert score_occasions(page, CONFIG)["Skiing"] == 0.0


def test_snow_rules_out_a_beach() -> None:
    page = [photo({"Beach": 0.7, "Shoreline": 0.7, "Snow": 0.93})]
    assert score_occasions(page, CONFIG)["Beach day"] == 0.0


def test_session_supplies_occasion_for_an_empty_page() -> None:
    page = [photo({"Person": 0.99, "Outdoors": 0.9})]
    session = [photo({"Beach": 0.9, "Shoreline": 0.8, "Outdoors": 0.9}) for _ in range(5)]
    result = understand_scene(page, context=session, config=CONFIG)
    assert result.occasion is not None
    assert result.occasion.name == "Beach day"
    assert result.source == "context"


def test_indoor_page_cannot_borrow_an_outdoor_occasion() -> None:
    page = [photo({"Person": 0.99, "Indoors": 0.95, "Living Room": 0.9})]
    session = [photo({"Beach": 0.9, "Shoreline": 0.8}) for _ in range(5)]
    assert understand_scene(page, context=session, config=CONFIG).occasion is None


def test_page_with_blocking_label_cannot_borrow() -> None:
    page = [photo({"Person": 0.99, "Snow": 0.9})]
    session = [photo({"Beach": 0.9, "Shoreline": 0.8}) for _ in range(5)]
    assert understand_scene(page, context=session, config=CONFIG).occasion is None


def test_one_odd_label_does_not_veto_the_session() -> None:
    page = [photo({"Person": 0.99})]
    session = [photo({"Skiing": 0.9, "Piste": 0.9, "Snow": 0.9}) for _ in range(5)]
    session.append(photo({"Beach": 0.7, "Snow": 0.9}))
    result = understand_scene(page, context=session, config=CONFIG)
    assert result.occasion is not None
    assert result.occasion.name == "Skiing"


def test_clear_session_overrules_weak_page_evidence() -> None:
    page = [photo({"Picnic": 0.7, "Picnic Basket": 0.6})]
    session = [photo({"Party Hat": 0.9, "Birthday Party": 0.9}) for _ in range(5)]
    result = understand_scene(page, context=session, config=CONFIG)
    assert result.occasion is not None
    assert result.occasion.name == "Birthday party"


def test_strong_page_evidence_beats_the_session() -> None:
    page = [photo({"Skiing": 0.95, "Piste": 0.95, "Snow": 0.95})]
    session = [photo({"Party Hat": 0.9, "Birthday Party": 0.9}) for _ in range(5)]
    result = understand_scene(page, context=session, config=CONFIG)
    assert result.occasion is not None
    assert result.occasion.name == "Skiing"
    assert result.source == "page"


def test_page_setting() -> None:
    assert page_setting([photo({"Indoors": 0.9, "Living Room": 0.8})]) == "indoor"
    assert page_setting([photo({"Outdoors": 0.9, "Nature": 0.8})]) == "outdoor"
    assert page_setting([photo({"Person": 0.9}, datetime(2020, 1, 1))]) == "unknown"
