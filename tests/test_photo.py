from datetime import datetime

import pytest

from captioner.photo import load_photo, parse_label_file, photos_in, read_exif

from .conftest import MakePhoto


def test_parse_label_file_handles_unquoted_keys(make_photo: MakePhoto) -> None:
    path = make_photo(labels={"Party Hat": 0.9, "Person": 0.99})
    labels = parse_label_file(f"{path}.json")
    assert [(lab.name, lab.confidence) for lab in labels] == [("Party Hat", 0.9), ("Person", 0.99)]


def test_read_exif_date_and_gps(make_photo: MakePhoto) -> None:
    path = make_photo(taken_at=datetime(2019, 2, 7, 16, 8, 23), lat=51.4576, lon=-0.0356)
    taken_at, lat, lon = read_exif(path)
    assert taken_at == datetime(2019, 2, 7, 16, 8, 23)
    assert lat == pytest.approx(51.4576, abs=1e-4)
    assert lon == pytest.approx(-0.0356, abs=1e-4)


def test_read_exif_southern_and_eastern_hemisphere(make_photo: MakePhoto) -> None:
    _, lat, lon = read_exif(make_photo(lat=-33.8912, lon=151.2777))
    assert lat == pytest.approx(-33.8912, abs=1e-4)
    assert lon == pytest.approx(151.2777, abs=1e-4)


def test_read_exif_without_metadata(make_photo: MakePhoto) -> None:
    assert read_exif(make_photo()) == (None, None, None)


def test_zero_gps_is_treated_as_missing(make_photo: MakePhoto) -> None:
    _, lat, lon = read_exif(make_photo(lat=0.0, lon=0.0))
    assert lat is None
    assert lon is None


def test_load_photo_missing_file_gives_empty_photo() -> None:
    p = load_photo("does/not/exist.jpg")
    assert p.taken_at is None
    assert not p.has_gps
    assert p.labels == ()


def test_load_photo_without_label_file(make_photo: MakePhoto) -> None:
    p = load_photo(str(make_photo(taken_at=datetime(2020, 1, 1))))
    assert p.labels == ()
    assert p.taken_at == datetime(2020, 1, 1)


def test_confidences_filters_by_threshold(make_photo: MakePhoto) -> None:
    p = load_photo(str(make_photo(labels={"Beach": 0.9, "Sea": 0.4})))
    assert p.confidences(0.5) == {"Beach": pytest.approx(0.9)}


def test_photos_in_skips_other_files_and_sorts_by_time(make_photo: MakePhoto) -> None:
    late = make_photo(taken_at=datetime(2020, 2, 7, 12), name="a.jpg")
    early = make_photo(taken_at=datetime(2020, 2, 7, 9), name="b.jpg")
    undated = make_photo(name="c.jpg")
    (late.parent / "notes.txt").write_text("not a photo")
    assert photos_in(late.parent) == [str(early), str(late), str(undated)]
