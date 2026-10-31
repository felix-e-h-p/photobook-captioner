from datetime import datetime

import pytest

from captioner import caption_page, generate_caption
from captioner.facts import PageFacts

from .conftest import KNUTSFORD, YUZAWA, MakePhoto


def test_end_to_end_one_off_occasion(make_photo: MakePhoto) -> None:
    page = [
        make_photo(
            taken_at=datetime(2018, 9, 14, 11),
            labels={"Cookie": 0.9, "Biscuit": 0.9, "Oven": 0.8, "Indoors": 0.9},
            **KNUTSFORD,
        )
    ]
    assert generate_caption([str(p) for p in page]) == "14th September 2018 – Baking"


def test_end_to_end_trip_abroad(make_photo: MakePhoto) -> None:
    page = [
        make_photo(taken_at=datetime(2020, 2, 7), labels={"Skiing": 0.9, "Piste": 0.9}, **YUZAWA)
        for _ in range(2)
    ]
    caption = generate_caption([str(p) for p in page])
    assert caption == "Skiing holiday in Yuzawa, Japan, February 2020"


def test_album_photos_from_same_session_help(make_photo: MakePhoto) -> None:
    when = datetime(2018, 9, 13, 12)
    page = [str(make_photo(taken_at=when, labels={"Person": 0.99}, **KNUTSFORD))]
    album = [
        str(make_photo(taken_at=when, labels={"Cookie": 0.9, "Oven": 0.9}, **KNUTSFORD))
        for _ in range(4)
    ]
    assert generate_caption(page) == "Knutsford, September 2018"
    result = caption_page(page, album_paths=album)
    assert result.caption == "13th September 2018 – Baking"
    assert result.n_context == 4
    assert "occasion borrowed from same-session photos" in result.facts.warnings


def test_album_photos_from_another_day_are_ignored(make_photo: MakePhoto) -> None:
    page = [str(make_photo(taken_at=datetime(2018, 9, 13), labels={"Person": 0.9}, **KNUTSFORD))]
    album = [str(make_photo(taken_at=datetime(2018, 12, 25), labels={"Cookie": 0.9}, **KNUTSFORD))]
    assert caption_page(page, album_paths=album).n_context == 0


def test_no_metadata_and_no_labels(make_photo: MakePhoto) -> None:
    assert generate_caption([str(make_photo())]) == "Memories"


def test_missing_file_does_not_crash() -> None:
    assert generate_caption(["not/a/photo.jpg"]) == "Memories"


@pytest.mark.parametrize("n", [0, 4])
def test_page_size_is_checked(make_photo: MakePhoto, n: int) -> None:
    with pytest.raises(ValueError, match="1 to 3 photos"):
        generate_caption([str(make_photo()) for _ in range(n)])


class _InventingWriter:
    name = "inventing"

    def write(self, facts: PageFacts) -> str:
        return "Summer in Cornwall, July 2023"


class _GoodWriter:
    name = "good"

    def write(self, facts: PageFacts) -> str:
        return "Ski week in Yuzawa, Feb 2020"


def test_writer_that_invents_facts_falls_back_to_template(make_photo: MakePhoto) -> None:
    page = [str(make_photo(taken_at=datetime(2020, 2, 7), labels={"Skiing": 0.9}, **YUZAWA))]
    result = caption_page(page, writer=_InventingWriter())
    assert result.caption == result.template_caption
    assert result.rejected


def test_writer_that_sticks_to_the_facts_is_kept(make_photo: MakePhoto) -> None:
    page = [str(make_photo(taken_at=datetime(2020, 2, 7), labels={"Skiing": 0.9}, **YUZAWA))]
    result = caption_page(page, writer=_GoodWriter())
    assert result.caption == "Ski week in Yuzawa, Feb 2020"
    assert result.rejected == []
