"""LLM writer tests."""

import json
from datetime import datetime
from typing import Any

import pytest

from captioner import LLMWriter, TemplateWriter, WriterError, caption_page, writer_from_env
from captioner.llm import add_date, facts_for_prompt

from .conftest import YUZAWA, MakePhoto, page_facts


class StubLLM(LLMWriter):
    """LLMWriter with the model call replaced by a fixed reply."""

    def __init__(self, reply: str | Exception) -> None:
        super().__init__(model="stub")
        self.reply = reply
        self.seen: list[dict[str, str]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.seen = messages
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _page(make_photo: MakePhoto) -> list[str]:
    return [str(make_photo(taken_at=datetime(2020, 2, 7), labels={"Skiing": 0.9}, **YUZAWA))]


def test_prompt_contains_facts_only(make_photo: MakePhoto) -> None:
    page = _page(make_photo)
    writer = StubLLM("Ski week in Yuzawa, Feb 2020")
    caption_page(page, writer=writer)
    sent = json.dumps(writer.seen)
    assert page[0] not in sent
    assert "Yuzawa, Japan" in sent
    assert "2020" not in sent


def test_good_caption_is_kept(make_photo: MakePhoto) -> None:
    result = caption_page(_page(make_photo), writer=StubLLM("Ski week in Yuzawa, Feb 2020"))
    assert result.caption == "Ski week in Yuzawa, Feb 2020"
    assert result.rejected == []


def test_invented_place_falls_back_to_template(make_photo: MakePhoto) -> None:
    result = caption_page(_page(make_photo), writer=StubLLM("Powder days in Niseko, Feb 2020"))
    assert result.caption == result.template_caption
    assert result.writer_caption == "Powder days in Niseko, Feb 2020"
    assert any("Niseko" in r for r in result.rejected)


def test_failure_falls_back_to_template(make_photo: MakePhoto) -> None:
    result = caption_page(_page(make_photo), writer=StubLLM(WriterError("timed out")))
    assert result.caption == result.template_caption
    assert result.rejected == ["writer failed: timed out"]


def test_ollama_not_running_falls_back_to_template(make_photo: MakePhoto) -> None:
    writer = LLMWriter(host="http://127.0.0.1:9", timeout=1)
    result = caption_page(_page(make_photo), writer=writer)
    assert result.caption == result.template_caption
    assert result.rejected[0].startswith("writer failed")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('"Ski week in Yuzawa."', "Ski week in Yuzawa, February 2020"),
        ("Ski week in Yuzawa\nHere is another option", "Ski week in Yuzawa, February 2020"),
        ("  “Ski week”  ", "Ski week, February 2020"),
    ],
)
def test_reply_is_cleaned(reply: str, expected: str) -> None:
    assert StubLLM(reply).write(page_facts()) == expected


def test_date_is_added_by_code() -> None:
    assert StubLLM("Ski week in Yuzawa").write(page_facts()) == "Ski week in Yuzawa, February 2020"


def test_one_off_occasion_puts_the_date_first() -> None:
    f = page_facts(date_text="7th February 2019", date_style="day")
    assert add_date("Party in the garden", f) == "7th February 2019 – Party in the garden"


def test_long_titles_get_a_short_month() -> None:
    title = "Family beach day at North Bondi in New South Wales"
    assert add_date(title, page_facts(date_text="March 2018")) == f"{title}, Mar 2018"


def test_date_from_the_model_is_left_alone() -> None:
    assert add_date("Ski week, Feb 2020", page_facts()) == "Ski week, Feb 2020"


def test_empty_reply_is_an_error() -> None:
    with pytest.raises(WriterError):
        StubLLM("   ").write(page_facts())


def test_prompt_facts_respect_use_place() -> None:
    sent: dict[str, Any] = facts_for_prompt(page_facts(use_place=False))
    assert sent["place"] is None


def test_writer_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAPTION_WRITER", raising=False)
    assert isinstance(writer_from_env(), TemplateWriter)
    monkeypatch.setenv("CAPTION_WRITER", "llm")
    assert isinstance(writer_from_env(), LLMWriter)
    monkeypatch.setenv("CAPTION_WRITER", "gpt")
    with pytest.raises(ValueError, match="CAPTION_WRITER"):
        writer_from_env()


def test_no_occasion_skips_the_model() -> None:
    writer = StubLLM("Autumn walk in the woods")
    f = page_facts(occasion=None, phrase=None, place="Knutsford", date_text="September 2018")
    assert writer.write(f) == "Knutsford, September 2018"
    assert writer.seen == []
