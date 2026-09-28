"""
Reads results/benchmarks.csv (written by stage2_fast/benchmark.cpp) and
renders a speedup chart to results/speedup_chart.png for the README.

Run after stage2_benchmark.exe: `python tools/make_chart.py`
"""

import csv
import os

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
CSV_PATH = os.path.join(RESULTS_DIR, "benchmarks.csv")
OUT_PATH = os.path.join(RESULTS_DIR, "speedup_chart.png")

LABELS = {
    ("naive", "1"): "naive",
    ("cache_friendly", "1"): "cache-friendly",
    ("threaded", "2"): "threaded (2t)",
    ("threaded", "4"): "threaded (4t)",
}

BLUE = "#2a78d6"
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def label_for(variant, threads):
    key = (variant, threads)
    if key in LABELS:
        return LABELS[key]
    return f"threaded ({threads}t)"


def main():
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # de-dupe/keep last "threaded" row with the highest thread count as the
    # headline "max threads" bar (avoids double-plotting 2t/4t/12t if the
    # hardware_concurrency happens to equal 2 or 4)
    seen_labels = {}
    for row in rows:
        label = label_for(row["variant"], row["threads"])
        seen_labels[label] = float(row["time_ms"])  # last write wins

    labels = list(seen_labels.keys())
    times = list(seen_labels.values())
    baseline = times[0]
    speedups = [baseline / t for t in times]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    bars = ax.bar(labels, times, color=BLUE, width=0.55, zorder=3)

    ax.set_yscale("log")
    ax.set_ylabel("time (ms, log scale)", color=SECONDARY_INK)
    ax.set_title("512x512 matmul: naive vs. cache-friendly vs. threaded", color=PRIMARY_INK,
                 fontsize=13, pad=16)

    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", colors=SECONDARY_INK)
    ax.tick_params(axis="y", colors=MUTED)

    for bar, t, s in zip(bars, times, speedups):
        label = f"{t:.1f} ms" if s == 1.0 else f"{t:.1f} ms ({s:.0f}x)"
        ax.text(bar.get_x() + bar.get_width() / 2, t * 1.15, label, ha="center", va="bottom",
                color=PRIMARY_INK, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
