"""Evaluation - sample pages, caption them, score the captions."""

from .metrics import PageRecord, summarise
from .run import evaluate
from .sampling import dedupe, sample_pages

__all__ = ["PageRecord", "dedupe", "evaluate", "sample_pages", "summarise"]
