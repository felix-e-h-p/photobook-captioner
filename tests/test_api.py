from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from captioner.api import app

from .conftest import YUZAWA, MakePhoto

client = TestClient(app)


@pytest.fixture
def ski_photo(make_photo: MakePhoto) -> Path:
    return make_photo(taken_at=datetime(2020, 2, 7), labels={"Skiing": 0.9}, **YUZAWA)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_caption_upload_with_labels(ski_photo: Path) -> None:
    label_file = Path(f"{ski_photo}.json")
    files = [
        ("photos", (ski_photo.name, ski_photo.read_bytes(), "image/jpeg")),
        ("labels", (label_file.name, label_file.read_bytes(), "application/json")),
    ]
    response = client.post("/caption", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["caption"] == "Skiing holiday in Yuzawa, Japan, February 2020"
    assert body["occasion"] == "Skiing"


def test_caption_upload_without_labels_falls_back_to_place_and_date(ski_photo: Path) -> None:
    files = [("photos", (ski_photo.name, ski_photo.read_bytes(), "image/jpeg"))]
    response = client.post("/caption", files=files)
    assert response.json()["caption"] == "Yuzawa, Japan, February 2020"


def test_caption_upload_rejects_more_than_three_photos(ski_photo: Path) -> None:
    files = [("photos", (f"{i}.jpg", ski_photo.read_bytes(), "image/jpeg")) for i in range(4)]
    assert client.post("/caption", files=files).status_code == 422


def test_caption_paths(ski_photo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAPTIONER_PHOTO_ROOT", str(ski_photo.parent))
    response = client.post("/caption/paths", json={"images": [str(ski_photo)]})
    assert response.status_code == 200
    assert response.json()["caption"] == "Skiing holiday in Yuzawa, Japan, February 2020"


def test_caption_paths_refuses_files_outside_the_photo_root(
    ski_photo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPTIONER_PHOTO_ROOT", str(ski_photo.parent / "library"))
    response = client.post("/caption/paths", json={"images": [str(ski_photo)]})
    assert response.status_code == 403


def test_unknown_writer_is_rejected(ski_photo: Path) -> None:
    files = [("photos", (ski_photo.name, ski_photo.read_bytes(), "image/jpeg"))]
    assert client.post("/caption?writer=gpt", files=files).status_code == 422
