import json
from datetime import datetime
from pathlib import Path

import pytest

from captioner.cli import main

from .conftest import KNUTSFORD, MakePhoto

BAKING = {"Cookie": 0.9, "Oven": 0.9, "Indoors": 0.9}


def test_caption_one_page(make_photo: MakePhoto, capsys: pytest.CaptureFixture[str]) -> None:
    path = make_photo(taken_at=datetime(2018, 9, 14), labels=BAKING, **KNUTSFORD)
    assert main(["caption", str(path)]) == 0
    assert capsys.readouterr().out.startswith("14th September 2018 – Baking")


def test_caption_json(make_photo: MakePhoto, capsys: pytest.CaptureFixture[str]) -> None:
    path = make_photo(taken_at=datetime(2018, 9, 14), labels=BAKING, **KNUTSFORD)
    main(["caption", str(path), "--json"])
    result = json.loads(capsys.readouterr().out)
    assert result["facts"]["occasion"] == "Baking"


def test_folder_writes_jsonl(make_photo: MakePhoto, tmp_path: Path) -> None:
    for hour in range(5):
        make_photo(taken_at=datetime(2018, 9, 14, 10 + hour), labels=BAKING, **KNUTSFORD)
    out = tmp_path / "out" / "captions.jsonl"
    out.parent.mkdir()
    assert main(["folder", str(tmp_path), "--page-size", "2", "--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 3
    assert [len(json.loads(line)["images"]) for line in lines] == [2, 2, 1]


def test_too_many_photos_is_an_error(
    make_photo: MakePhoto, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = [str(make_photo()) for _ in range(4)]
    assert main(["caption", *paths]) == 2
    assert "1 to 3 photos" in capsys.readouterr().err


def test_empty_folder_is_an_error(tmp_path: Path) -> None:
    assert main(["folder", str(tmp_path)]) == 1
