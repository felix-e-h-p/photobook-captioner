"""Captions for photobook pages from photo metadata and scene labels."""

from .llm import LLMWriter, writer_from_env
from .pipeline import PageResult, caption_page, generate_caption
from .writers import TemplateWriter, WriterError, validate

__all__ = [
    "LLMWriter",
    "PageResult",
    "TemplateWriter",
    "WriterError",
    "caption_page",
    "generate_caption",
    "validate",
    "writer_from_env",
]
