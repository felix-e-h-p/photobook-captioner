"""Figures for the write-up, from the final MLflow runs and the hand review.

    uv run python scripts/figures.py

Writes PNGs to docs/figures/. Same run selection as notebooks/03_results.ipynb.
"""

import json
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import OmegaConf

from captioner import TemplateWriter, caption_page
from captioner.config import load_config
from captioner.evaluation.metrics import page_record, summarise
from captioner.evaluation.sampling import dedupe, sample_pages
from captioner.photo import photos_in

FINAL_FROM = pd.Timestamp(os.environ.get("FINAL_FROM", "2026-10-31 18:39"), tz="UTC")
REVIEW_RUN = os.environ.get("REVIEW_RUN", "181aa8962a0746869bfef67eaac47662")
OUT = Path("docs/figures")

TEMPLATE, LLM, PAGE_ONLY, BORROWED = "#2a78d6", "#eb6834", "#9a9a9a", "#1baf7a"
INK, MUTED = "#222222", "#666666"
FOLDERS = {
    "day-at-beach": "beach",
    "garden-birthday-party": "party",
    "skiing-holiday": "ski",
    "summer-with-grandkids": "grandkids",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#bbbbbb",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def load_runs() -> pd.DataFrame:
    runs = mlflow.search_runs(experiment_names=["caption-evaluation"], order_by=["start_time DESC"])
    runs["writer"] = runs["params.writer"].str.split(":").str[0]
    runs["judged"] = runs["params.judge"] != "off"
    return runs


def pick(writer: str, context: bool = True, judged: bool = False) -> pd.Series:
    runs = load_runs()
    match = runs[
        (runs["writer"] == writer)
        & (runs["params.use_context"] == str(context))
        & (runs["params.mode"] == "random")
        & (runs["judged"] == judged)
        & (runs["start_time"] >= FINAL_FROM)
    ]
    return match.iloc[0]


def pages_of(run_id: str) -> pd.DataFrame:
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="pages.jsonl")
    return pd.DataFrame([json.loads(line) for line in Path(path).read_text().splitlines() if line])


def bootstrap(values: np.ndarray, stat: Callable[[np.ndarray], float], n: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    return np.percentile([stat(values[i]) for i in idx], [2.5, 97.5])


def save(fig: Figure, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def context_by_page_size() -> None:
    """Dumbbell - accuracy page only against with context, per folder and page size."""
    page_only = pages_of(pick("template", context=False)["run_id"])
    context = pages_of(pick("template")["run_id"])
    labelled = context["expected"].notna()
    right = context["outcome"] == "right"
    right_alone = page_only["outcome"] == "right"
    rows = []
    for _, group in context[labelled].groupby(["folder", "n_photos"]):
        folder, k = group["folder"].iloc[0], int(group["n_photos"].iloc[0])
        name = f"{FOLDERS[folder]}, {k} photo{'s' if k > 1 else ''}"
        rows.append((name, right_alone[group.index].mean(), right[group.index].mean()))
    for k in (1, 2, 3):
        idx = context[labelled & (context["n_photos"] == k)].index
        name = f"all, {k} photo{'s' if k > 1 else ''}"
        rows.append((name, right_alone[idx].mean(), right[idx].mean()))
    data = pd.DataFrame(rows, columns=["group", "page only", "with context"])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.arange(len(data))[::-1]
    ax.hlines(y, data["page only"], data["with context"], color="#cfcfcf", lw=2.5, zorder=1)
    ax.scatter(data["page only"], y, s=42, color=PAGE_ONLY, label="page only", zorder=2)
    ax.scatter(
        data["with context"], y, s=42, color=TEMPLATE, label="with session context", zorder=3
    )
    ax.axhline(3 - 0.5, color="#bbbbbb", lw=0.8)
    ax.set_yticks(y, data["group"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("occasion accuracy")
    ax.grid(axis="y", visible=False)
    ax.set_title("Session context matters most for single-photo pages", pad=24)
    ax.legend(loc="lower left", ncols=2, bbox_to_anchor=(0, 1.0), borderaxespad=0, fontsize=8)
    fig.text(
        0,
        -0.06,
        "Template writer, 30 pages per folder and page size. Grandkids has no expected occasion.",
        color=MUTED,
        fontsize=7.5,
    )
    save(fig, "context_by_page_size.png")


def writer_trade_off() -> None:
    """Variety against judge faithfulness, per folder, template to LLM, with bootstrap intervals."""
    template, llm = pick("template", judged=True), pick("llm", judged=True)
    tp, lp = pages_of(template["run_id"]), pages_of(llm["run_id"])

    def faithful(df: pd.DataFrame) -> np.ndarray:
        return np.array([j["faithful"] if j else np.nan for j in df["judge"]], dtype=float)

    def distinct(captions: np.ndarray) -> float:
        return len(set(captions)) / len(captions)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for folder, short in FOLDERS.items():
        start = (
            template[f"metrics.{folder}/distinct_captions"],
            template[f"metrics.{folder}/judge_faithful"],
        )
        end = (llm[f"metrics.{folder}/distinct_captions"], llm[f"metrics.{folder}/judge_faithful"])
        ax.annotate(
            "", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#c8c8c8", "lw": 1.2}
        )
        ax.scatter(*start, s=22, color=TEMPLATE, alpha=0.45, zorder=2)
        ax.scatter(*end, s=22, color=LLM, alpha=0.45, zorder=2)
        ax.annotate(
            short, end, xytext=(5, -2), textcoords="offset points", fontsize=7.5, color=MUTED
        )

    for df, colour, name in [(tp, TEMPLATE, "template"), (lp, LLM, "LLM")]:
        f = faithful(df)
        f = f[~np.isnan(f)]
        captions = df["caption"].to_numpy()
        x, y = distinct(captions), f.mean()
        xerr = np.abs(bootstrap(captions, distinct) - x)[:, None]
        yerr = np.abs(bootstrap(f, np.mean) - y)[:, None]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            ms=8,
            color=colour,
            ecolor=colour,
            elinewidth=1.5,
            capsize=3,
            zorder=4,
            label=f"{name}, all folders",
        )

    ax.set_xlabel("distinct captions (share of pages)")
    ax.set_ylabel("judge faithfulness (1 to 5)")
    ax.set_xlim(0, None)
    ax.set_title("LLM buys variety at a small cost in faithfulness")
    ax.legend(loc="center right", fontsize=8)
    fig.text(
        0,
        -0.06,
        "95% bootstrap intervals over pages. Faded points per folder, template to LLM.",
        color=MUTED,
        fontsize=7.5,
    )
    save(fig, "writer_trade_off.png")


def hand_against_judge() -> None:
    """5x5 agreement matrix - hand faithfulness against judge faithfulness on the same pages."""
    review = pd.read_csv(Path("outputs/review") / REVIEW_RUN / "review.csv")
    pages = pages_of(REVIEW_RUN)
    pages["key"] = pages["images"].str.join(";")
    judge = pd.json_normalize(pages["judge"]).set_index(pages["key"])
    both = review.join(judge[~judge.index.duplicated()]["faithful"].rename("judge"), on="images")
    counts = pd.crosstab(both["judge"], both["faithful"]).reindex(
        index=range(1, 6), columns=range(1, 6), fill_value=0
    )

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.imshow(counts.values, cmap="Blues", origin="lower", vmin=0)
    for (i, j), v in np.ndenumerate(counts.values):
        if v:
            ax.text(
                j,
                i,
                v,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if v > counts.values.max() / 2 else INK,
            )
    ax.plot([-0.5, 4.5], [-0.5, 4.5], color="#888888", lw=0.8, ls="--")
    ax.set_xticks(range(5), [str(i) for i in range(1, 6)])
    ax.set_yticks(range(5), [str(i) for i in range(1, 6)])
    ax.set_xlabel("hand faithfulness")
    ax.set_ylabel("judge faithfulness")
    ax.grid(False)
    ax.set_title("Judge is more lenient than a person")
    fig.text(
        0,
        -0.06,
        f"{len(both)} pages, same captions. Above the diagonal, judge scored higher.",
        color=MUTED,
        fontsize=7.5,
    )
    save(fig, "hand_against_judge.png")


def session_window() -> None:
    """Accuracy and over-reach as the session window grows. Re-runs the template, ~10 seconds."""
    expected = cast(
        dict[str, Any], OmegaConf.to_container(OmegaConf.load("configs/evaluate.yaml").expected)
    )
    base, writer = load_config(), TemplateWriter()
    albums = {f: photos_in(Path("data") / f) for f in sorted(expected)}
    pages = {
        f: sample_pages(dedupe(a), (1, 2, 3), 30, "random", seed)
        for seed, (f, a) in enumerate(albums.items())
    }
    windows = [0.5, 1, 2, 4, 6, 12, 24, 48, 72]
    rows = []
    for hours in windows:
        config = replace(base, settings=replace(base.settings, context_window_hours=hours))
        records = [
            page_record(caption_page(p, albums[f], writer, config=config), f, expected[f])
            for f, ps in pages.items()
            for p in ps
        ]
        m = summarise(records)
        rows.append((hours, m["all/accuracy"], m["summer-with-grandkids/occasion_borrowed"]))
    data = pd.DataFrame(rows, columns=["hours", "accuracy", "borrowed"])

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(data["hours"], data["accuracy"], marker="o", ms=4, lw=2, color=TEMPLATE)
    ax.plot(data["hours"], data["borrowed"], marker="o", ms=4, lw=2, color=BORROWED)
    ax.annotate(
        "accuracy, labelled folders",
        (72, data["accuracy"].iloc[-1]),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        color=TEMPLATE,
        fontsize=8,
    )
    ax.annotate(
        "grandkids pages borrowing an occasion",
        (72, data["borrowed"].iloc[-1]),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        color=BORROWED,
        fontsize=8,
    )
    ax.axvline(6, color="#888888", lw=0.8, ls="--")
    ax.annotate(
        "default, 6 h",
        (6, 0.04),
        xytext=(4, 0),
        textcoords="offset points",
        color=MUTED,
        fontsize=7.5,
    )
    ax.set_xscale("log")
    ax.set_xticks(windows, [f"{w:g}" for w in windows])
    ax.minorticks_off()
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("session window (hours, log scale)")
    ax.set_ylabel("share of pages")
    ax.set_title("Longer sessions borrow more, but accuracy can't see it")
    fig.text(
        0,
        -0.06,
        "Labelled folders are single events, hence accuracy stays flat. Grandkids spans a weekend.",
        color=MUTED,
        fontsize=7.5,
    )
    save(fig, "session_window.png")


def repetition() -> None:
    """Share of pages carrying a folder's most common caption, book layout, both writers."""

    def top_share(writer: str) -> pd.Series:
        path = Path("outputs/examples") / writer / "captions.jsonl"
        rows = pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line])
        return rows.groupby("folder")["caption"].agg(
            lambda c: c.value_counts(normalize=True).iloc[0]
        )

    data = pd.DataFrame({"template": top_share("template"), "LLM": top_share("llm")}).rename(
        index=FOLDERS
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(data))
    width = 0.36
    for offset, (name, colour) in zip(
        (-width / 2, width / 2), [("template", TEMPLATE), ("LLM", LLM)], strict=True
    ):
        bars = ax.bar(x + offset, data[name], width - 0.04, color=colour, label=name)
        ax.bar_label(
            bars, labels=[f"{v:.0%}" for v in data[name]], padding=2, fontsize=8, color=INK
        )
    ax.set_xticks(x, data.index)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", visible=False)
    ax.set_title("Laid out as a book, one caption dominates each folder")
    ax.legend(loc="upper right", ncols=2, fontsize=8)
    fig.text(
        0,
        -0.06,
        "Share of pages with the folder's most common caption. 2 photos per page, time order.",
        color=MUTED,
        fontsize=7.5,
    )
    save(fig, "repetition.png")


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    OUT.mkdir(parents=True, exist_ok=True)
    context_by_page_size()
    writer_trade_off()
    hand_against_judge()
    session_window()
    repetition()
