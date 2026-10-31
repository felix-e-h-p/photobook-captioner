"""Judge and CLIP tests. No models are called - replies and scores are stubbed."""

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pytest

from captioner import TemplateWriter
from captioner.evaluation import evaluate, summarise
from captioner.evaluation.clip import strip_date
from captioner.evaluation.judge import Judge, scene_labels
from captioner.ollama import OllamaError, chat

from .conftest import YUZAWA, MakePhoto, photo


class StubJudge(Judge):
    def __init__(self, reply: str | Exception) -> None:
        super().__init__(model="stub")
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class StubClip:
    def similarity(self, images: Sequence[str], text: str) -> float:
        return 0.3 if "Ski" in text else 0.2


def test_judge_parses_and_clamps_scores() -> None:
    reply = json.dumps({"faithful": 9, "specific": 0, "natural": 4, "unsupported": ["attic"]})
    verdict = StubJudge(reply).judge("Kids in the attic", {"place": "Knutsford"})
    assert verdict is not None
    assert (verdict.faithful, verdict.specific, verdict.natural) == (5, 1, 4)
    assert verdict.unsupported == ["attic"]


@pytest.mark.parametrize("reply", ["not json", '{"faithful": 5}', OllamaError("down")])
def test_judge_failure_gives_no_verdict(reply: str | Exception) -> None:
    assert StubJudge(reply).judge("Caption", {}) is None


def test_scene_labels_skip_generic_and_weak_labels() -> None:
    page = [photo({"Person": 0.99, "Cookie": 0.9, "Oven": 0.8, "Bread": 0.6})]
    assert scene_labels(page) == ["Cookie", "Oven"]


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


def test_evaluate_adds_judge_and_clip_scores(make_photo: MakePhoto, tmp_path: Path) -> None:
    (tmp_path / "ski").mkdir()
    for i in range(3):
        make_photo(
            taken_at=datetime(2020, 2, 7, 10 + i),
            labels={"Skiing": 0.9, "Piste": 0.9},
            name=f"ski/{i}.jpg",
            **YUZAWA,
        )
    reply = json.dumps({"faithful": 5, "specific": 4, "natural": 3, "unsupported": []})
    judge = StubJudge(reply)
    records = evaluate(
        tmp_path,
        {"ski": "Skiing"},
        TemplateWriter(),
        sizes=(1,),
        pages_per_size=2,
        remove_duplicates=False,
        judge=judge,
        clip=StubClip(),
    )
    metrics = summarise(records)
    assert metrics["all/judge_faithful"] == 5
    assert metrics["all/judge_unsupported"] == 0
    assert metrics["all/clip"] == 0.3
    assert metrics["all/clip_gain"] == 0
    assert "Skiing" in json.loads(judge.prompts[0])["known"]["detected_in_photos"]


def test_chat_raises_when_ollama_is_not_running() -> None:
    with pytest.raises(OllamaError):
        chat([{"role": "user", "content": "hi"}], "any", host="http://127.0.0.1:9", timeout=1)
