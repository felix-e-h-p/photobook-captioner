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
