"""CLIP image-text similarity - does the caption describe what's actually in the photos?"""

from collections.abc import Sequence
from typing import Any, Final, Protocol

from ..dates import strip_date

DEFAULT_MODEL: Final = "ViT-B-32"
DEFAULT_PRETRAINED: Final = "laion2b_s34b_b79k"


class ImageTextScorer(Protocol):
    def similarity(self, images: Sequence[str], text: str) -> float: ...


class ClipScorer:
    """Cosine similarity between the caption and the mean of the page's image embeddings."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | None = None,
    ) -> None:
        import open_clip
        import torch

        self._torch: Any = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained
        )
        self.model: Any = clip_model.to(self.device).eval()
        self.preprocess: Any = preprocess
        self.tokenizer: Any = open_clip.get_tokenizer(model)

    def similarity(self, images: Sequence[str], text: str) -> float:
        from PIL import Image, ImageOps

        torch = self._torch
        pixels = []
        for path in images:
            with Image.open(path) as im:
                pixels.append(self.preprocess(ImageOps.exif_transpose(im).convert("RGB")))
        with torch.no_grad():
            image = self.model.encode_image(torch.stack(pixels).to(self.device))
            image = image / image.norm(dim=-1, keepdim=True)
            image = image.mean(dim=0)
            image = image / image.norm()
            words = self.model.encode_text(self.tokenizer([strip_date(text)]).to(self.device))[0]
            words = words / words.norm()
        return round(float(image @ words), 4)
