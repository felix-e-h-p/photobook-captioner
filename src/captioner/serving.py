"""The captioner as an MLflow model, so it can be versioned, registered and served."""

from typing import Literal

from mlflow.models import set_model
from mlflow.pyfunc.model import PythonModel
from pydantic import BaseModel

from captioner.llm import make_writer
from captioner.pipeline import generate_caption


class Page(BaseModel):
    """One page to caption. `album` is optional - other photos from the same book."""

    images: list[str]
    album: list[str] | None = None
    home_country: str = "GB"


class CaptionModel(PythonModel):
    """Pages in, captions out. Set `writer` to "llm" to use the local LLM writer."""

    def predict(
        self,
        model_input: list[Page],
        params: dict[str, Literal["template", "llm"]] | None = None,
    ) -> list[str]:
        writer = make_writer((params or {}).get("writer", "template"))
        return [
            generate_caption(page.images, page.album, writer, page.home_country)
            for page in model_input
        ]


set_model(CaptionModel())
