# Photobook captioner - Write-up and analysis

## Summary

Captions for photobook pages of 1 to 3 photos, which are constructed from photo metadata and scene labels rather than from the pixels directly. The proposed pipeline extracts facts first (i.e. date, place, occasion), then words them either with a deterministic template by default, or by utilising a small local LLM for variety. Worth noting that it checks every LLM caption against the facts before it can be used.

The main results, for four sample folders:

- Session context appears to be the main driver of accuracy. Pages that may borrow evidence from photos taken around the same time reach approximately 0.98 occasion accuracy, against 0.59 on their own photos. Wrong occasions stay under 1%.
- The template and LLM writers are equally accurate. The LLM gives about 4 times the variety for a small yet relatively measurable cost in faithfulness.
- The main weakness is repetition across a book, whereby a single caption dominates each folder. Fix belongs above the page, captioning per section as opposed to per page.

Everything runs offline, at minimal cost, and no photo leaves the machine.

## 1. Exploration

Further detail situated in `notebooks/01_exploration.ipynb`.

Individual photos come with EXIF and a label file of scene labels with confidences. The capture dates and GPS are present for the majority of photos, hence date and place can be stated as facts. Labels are noisier - they name objects well, yet provide minimal information about people or occasions, and a single photo often carries no occasion specific label at all. A good example is that a close up of two people on a beach is labelled as people and clothing, not as a beach.

Folders contain bursts of almost identical photos, hence initial sampling removes near duplicates first (difference hash), so the same evidence isn't counted twice. Three folders are single events. Grandkids folder covers a weekend of different activities and has no single expected occasion, which makes it the useful stress test for anything that borrows evidence across photos.

The initial exploration stage led to three major design constraints. Facts that are reliable should be stated exactly. Occasion should only be named when evidence supports it. And a single page needs help from the photos situated around it.

## 2. Approaches considered

Options are directly compared and discussed in `notebooks/02_approaches.ipynb`:

- Vision-language model that captions photos directly. The most fluent, yet sees pixels only, hence it cannot know the date or the location, has a propensity to invent detail, and sends every photo to a model. Raises both cost and privacy concerns.
- Cloud LLM over labels and metadata. Grounded in facts, yet still an issue with per page API cost and latency, and still free to invent.
- Specified rules and a template over both labels and metadata. Grounded, free, instant and deterministic, but repetitive.

The chosen design is effectively facts first and then wording. Every claim in a caption traces back to EXIF or a label, runs offline, and the wording step is swappable. Template method is the default and a local LLM is the option for variety. Both pass the same validation criteria. A vision model can be implemented later as one more source of facts, behind the same validator, as opposed to acting as the writer.

The main guiding principle is that a printed caption can't be corrected. An invented detail is worse than a plain caption, hence the system abstains rather than guesses.

## 3. Design and implementation specifics

```
photos -> EXIF + labels -> page facts -> writer -> validator -> caption
```

Facts: Capture time and GPS come directly from EXIF. GPS is reverse geocoded offline to a location/town, with the country added when outside the user's home country. Labels are scored against occasion rules situated in `src/captioner/occasions.toml`, which hold label specific evidence weights, main labels that must be present, blocking labels that rule an occasion out, and expected setting alongside season.

Session context: A weak page may borrow an occasion from other photos taken within specified 6 hours and 25 km, yet only if session evidence is relatively clear, the page setting is compatible and nothing on the page blocks it. Strong page evidence will always override the session.

Writers: The template has two shapes, trips ("Skiing holiday in Yuzawa, Japan, February 2020") and single day occasions ("7th February 2019 – Birthday party"). The LLM writer (Llama 3.2 3B via Ollama) only ever sees a few words of text (occasion, place, season, setting and a few flags) and never the actual photos or dates. The date is added back afterwards. Pages with no occasion are set to skip the model entirely.

Validation: Every LLM caption is subsequently checked against the facts and hence rejected if it introduces years, months, days, proper nouns, people, or scene details that the facts do not support. Rejected captions will fall back to the template.

Packaging and tooling: The package is strictly typed with 140 tests. CI runs lint, type checks and tests, then builds the Docker image and checks API captions a photo. Captioner is available as a Python function (`generate_caption`), a CLI, an HTTP API and a versioned MLflow model. Evaluation runs are logged to MLflow with config, commit and outputs. Aim is to ensure full traceability.

## 4. Evaluation criteria

### Main results

| Run | Accuracy | Wrong | Abstained | Distinct captions | LLM rejected | Judge faithful |
|---|---|---|---|---|---|---|
| Template, page only | 0.59 | 2.2% | 39.3% | 5.6% | - | - |
| Template, with context | 0.99 | 0.4% | 1.1% | 3.9% | - | 4.80 |
| Template, with context, consecutive pages | 0.98 | 0.4% | 1.5% | 3.3% | - | - |
| LLM, page only | 0.59 | 2.2% | 39.3% | 14.4% | 3.9% | - |
| LLM, with context | 0.99 | 0.4% | 1.1% | 15.3% | 7.5% | 4.77 |
| LLM, with context, consecutive pages | 0.98 | 0.4% | 1.5% | 12.5% | 7.5% | - |

360 pages per run (270 with an expected occasion). Judge scored on context runs only. Manual review of 32 LLM captions gave 4.0 for faithfulness and 3.6 for naturalness, out of 5.
