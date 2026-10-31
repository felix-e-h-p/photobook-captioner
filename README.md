# Photobook captioner

[![CI](https://github.com/felix-e-h-p/photobook-captioner/actions/workflows/ci.yml/badge.svg)](https://github.com/felix-e-h-p/photobook-captioner/actions/workflows/ci.yml)

Captions for photobook pages of 1 to 3 photos, generated from both photo metadata and scene labels.

```
Family beach day in North Bondi, Australia, March 2018
7th February 2019 – Birthday party
Skiing holiday in Yuzawa, Japan, February 2020
13th–14th September 2018 – Baking with the kids
Knutsford, September 2018
```

More located in `outputs/examples/` - every folder laid out as a book, with both writers.

## The problem

Provided 1 to 3 photos on a page, the goal was to produce a short caption that a person would like to see printed in their own book. The caption should name what the occasion is where one is evident, indicate the date and location where known. Printed captions can't be corrected after, hence an invented detail could be considered technically worse than a caption lacking detail. When uncertainty exists, the approach says less rather than risk being incorrect.

## Method overview

Captioning is split into two fundamental stages - fact extraction and wording.

Fact extraction: Each photo provides capture time and GPS from EXIF, and scene labels with confidences from the label file alongside it. GPS is reverse geocoded offline to town, with country added when outside of home nation. Labels scored against occasion rules in `src/captioner/occasions.toml`, which includes per-label evidence weights, core labels, blocking labels, expected setting and season. Page score per occasion is mean over photos of weighted label confidence. Weak pages may borrow occasion from photos determined as in the same session (within 6 hours and 25 km), yet only where setting is deemed compatible and no blocking labels are present. Strong page evidence overrides session.

Wording: Default writer is a deterministic template with two shapes, trips ("Skiing holiday in Yuzawa, Japan, February 2020") and single day occasions ("7th February 2019 – Birthday party"). An optional writer rewords same facts with a small local LLM (Llama 3.2 3B via Ollama), for variety. The model only sees text facts, never photos or dates. Date is subsequently added back.

Validation: Every LLM caption is checked directly against facts. Will be rejected if it introduces years, months, days, proper nouns, people ("grandma", "solo") or scene details (weather, rooms, scenery) the facts do not support. Rejected captions fall back to the template.

    photos -> EXIF + labels -> page facts -> writer -> validator -> caption

Alternatives compared in `notebooks/02_approaches.ipynb` - a vision-language model captioning photos directly, a cloud LLM over labels, and the template alone. Facts first was chosen for grounding, cost and privacy. Every claim in a caption traces back to EXIF or a label, it runs offline at practically zero cost, and no photo is sent anywhere. A vision model fits in as one more source of facts situated behind the same validator.

## Evaluation

No reference captions exist, hence evaluation relies on checks that don't specifically need them.

- Occasion - folder names as weak labels. Each page scored right, wrong or abstained.
- Grounding - years in caption checked against EXIF, independently of validator. Town checked against GPS.
- Variety - share of distinct captions per folder, since a book repeating one caption reads badly.
- LLM judge - larger local model (Llama 3.1 8B) scores faithfulness, specificity and naturalness from 1 to 5, provided facts and detected labels.
- CLIP - image/text similarity of caption against photos (ViT-B/32), and gain over the template caption.
- Human review - stratified sample of 32 pages to manually check for faithfulness and naturalness.

Pages sampled per folder after near-duplicate removal (dHash), as random or consecutive photos. All runs logged to MLflow with config, commit and page specific outputs.

## Results

Session context is main driver of accuracy. Without it, approximately 40% of pages abstain. With it, occasion accuracy is approximately 0.98 on both random and consecutive pages, wrong occasions under 1%. Years in captions grounded in EXIF on every page.

Both the template and LLM writers are equally accurate on occasion, since both use same facts. LLM gives about 4 times the variety (15% distinct captions against 4%) at a small cost in faithfulness (judge 4.77 against 4.80 out of 5). Validator rejects about 7.5% of LLM captions, which fall back to the template. Adding scene words to the validator raised judge faithfulness of LLM captions from 4.64 to 4.75 in an earlier run, at some cost in variety. LLM variety moves a little between runs since sampling isn't deterministic, template is.

Manual review of 32 LLM captions, from a run pre scene/word fix, gave 4.0 for faithfulness and 3.6 for naturalness out of 5, approximately 69% and 47% of pages at 4 or above. Lowest scores came from invented scene details ("in the snow"), wrong labels and borrowed occasions.

Repetition here is the main weakness. Laid out as a book, 30 of 34 ski pages get the same template caption. LLM writer definitely helps, yet section specific captions would help more.

Full results, sample sizes and failure cases in `notebooks/03_results.ipynb` and the [write-up](docs/writeup.md). Example captions for every folder, both writers, in `outputs/examples/`.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Ollama only for LLM writer and judge (`ollama pull llama3.2:3b`, `ollama pull llama3.1:8b`).

Sample photo folders provided with the task to be situated in `data/`, each photo with its label file alongside, e.g. `data/skiing-holiday/Image_1.jpg` and `data/skiing-holiday/Image_1.jpg.json`.

    make install
    uv run captioner caption data/skiing-holiday/Image_1.jpg data/skiing-holiday/Image_2.jpg
    uv run captioner folder data/day-at-beach --page-size 2 --llm

From Python:

    from captioner import generate_caption

    generate_caption(["data/skiing-holiday/Image_1.jpg"])

Optional `album_paths` takes other photos in the book, so a page can draw on its session.

## Commands

    make check          ruff, strict mypy and tests - same as CI
    make evaluate       evaluate template writer, logged to MLflow
    make evaluate-llm   evaluate LLM writer with judge and CLIP
    make sweep          template and LLM, with and without session context
    make outputs        example captions for every folder, both writers
    make review RUN=id  rate a sample of captions by hand
    make score RUN=id   log ratings to the run
    make mlflow         compare runs in MLflow UI
    make serve          HTTP API, docs at /docs
    make docker         build API image
    make model          log and register captioner as an MLflow model

## Serving

API - `POST /caption` can take up to 3 photos with their label files, `POST /caption/paths` takes photos the server can read. Uploads exist only for length of the request.

Docker - two-stage build, non-root user, health check. CI builds image and then checks it captions a photo.

MLflow - pipeline logged as a versioned model, `photobook-captioner`. Needs dev dependencies.

## Code

    src/captioner/
        photo.py, geo.py, dates.py     EXIF, labels, places and dates
        scene.py, facts.py             labels to occasion, page facts
        writers.py, llm.py, ollama.py  template writer, validator, LLM writer
        pipeline.py                    photos in, caption out
        cli.py, api.py, serving.py     command line, HTTP API, MLflow model
        evaluation/                    sampling, metrics, LLM judge, CLIP
    notebooks/   exploration, approaches compared, results
    scripts/     evaluation, examples, human review, model logging
    configs/     evaluation settings (Hydra)
    outputs/     example captions and review ratings
    docs/        write-up
    tests/

## Privacy

Photos never leave the machine. Writers and judge see text facts only and no external API is called.

## Limitations

Data is small, with folder names as weak labels - hence results show behaviour, not production accuracy. Main weaknesses are repeated captions within an occasion, occasionally wrong scene labels, and pages borrowing an occasion that doesn't fit them. Details, examples and a prioritised list of fixes in the [write-up](docs/writeup.md).
