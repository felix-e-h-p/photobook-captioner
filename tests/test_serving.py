from datetime import datetime

from captioner.serving import CaptionModel, Page

from .conftest import YUZAWA, MakePhoto


def test_model_captions_each_page(make_photo: MakePhoto) -> None:
    path = make_photo(taken_at=datetime(2020, 2, 7), labels={"Skiing": 0.9}, **YUZAWA)
    pages = [Page(images=[str(path)]), Page(images=["missing.jpg"])]
    captions = CaptionModel().predict(pages)
    assert captions == ["Skiing holiday in Yuzawa, Japan, February 2020", "Memories"]
