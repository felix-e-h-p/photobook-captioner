from datetime import datetime
from pathlib import Path

from PIL import Image

from captioner import TemplateWriter
from captioner.evaluation import dedupe, evaluate, sample_pages, summarise
from captioner.evaluation.metrics import reason_kind

from .conftest import KNUTSFORD, YUZAWA, MakePhoto


def _striped(path: Path, vertical: bool) -> str:
    im = Image.new("L", (64, 64))
    for x in range(64):
        for y in range(64):
            band = (x if vertical else y) // 8
            im.putpixel((x, y), 255 if band % 2 else 0)
    im.save(path)
    return str(path)


def test_dedupe_keeps_one_photo_per_burst(tmp_path: Path) -> None:
    a = _striped(tmp_path / "a.jpg", vertical=True)
    b = _striped(tmp_path / "b.jpg", vertical=True)
    c = _striped(tmp_path / "c.jpg", vertical=False)
    assert sorted(Path(p).name for p in dedupe([a, b, c])) == ["a.jpg", "c.jpg"]


def test_random_pages_have_the_right_sizes() -> None:
    paths = [f"p{i}.jpg" for i in range(10)]
    pages = sample_pages(paths, sizes=(1, 3), pages_per_size=5, mode="random")
    assert [len(p) for p in pages] == [1] * 5 + [3] * 5


def test_consecutive_pages_are_neighbours_in_time(make_photo: MakePhoto) -> None:
    paths = [str(make_photo(taken_at=datetime(2020, 1, 1, h))) for h in range(6)]
    for page in sample_pages(paths, sizes=(3,), pages_per_size=10, mode="consecutive"):
        positions = [paths.index(p) for p in page]
        assert positions == list(range(positions[0], positions[0] + 3))


def test_pages_bigger_than_the_folder_are_skipped() -> None:
    assert sample_pages(["only.jpg"], sizes=(2,)) == []


def test_evaluate_scores_occasions_and_grounding(make_photo: MakePhoto, tmp_path: Path) -> None:
    for folder, labels, where in [
        ("ski", {"Skiing": 0.9, "Piste": 0.9}, YUZAWA),
        ("home", {"Person": 0.9}, KNUTSFORD),
    ]:
        (tmp_path / folder).mkdir()
        for i in range(4):
            make_photo(
                taken_at=datetime(2020, 2, 7, 10 + i),
                labels=labels,
                name=f"{folder}/{i}.jpg",
                **where,
            )
    records = evaluate(
        tmp_path,
        {"ski": "Skiing", "home": None},
        TemplateWriter(),
        sizes=(1, 2),
        pages_per_size=3,
        remove_duplicates=False,
    )
    metrics = summarise(records)
    assert len(records) == 12
    assert metrics["ski/accuracy"] == 1.0
    assert metrics["all/years_grounded"] == 1.0
    assert "home/accuracy" not in metrics
    assert {r.outcome for r in records if r.folder == "home"} == {"unlabelled"}


def test_rejection_reasons_are_grouped_by_kind() -> None:
    assert reason_kind("'Niseko' not supported by the facts") == "<word> not supported by the facts"
    assert reason_kind("year 2021 not in metadata") == "year N not in metadata"
    assert reason_kind("writer failed: timed out") == "writer failed"
