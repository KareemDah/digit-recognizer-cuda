"""
Reads results/cpu_benchmarks.csv (written by stage2_fast/benchmark.cpp) and
results/gpu_benchmarks.csv (hand-maintained, from Colab runs of
stage3_cuda/matmul.cu -- kept in a separate file specifically so re-running
the CPU benchmark can never overwrite it), and renders a combined speedup
chart to results/speedup_chart.png for the README.

Run after stage2_benchmark.exe: `python tools/make_chart.py`
"""

import csv
import os

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
CPU_CSV_PATH = os.path.join(RESULTS_DIR, "cpu_benchmarks.csv")
GPU_CSV_PATH = os.path.join(RESULTS_DIR, "gpu_benchmarks.csv")
OUT_PATH = os.path.join(RESULTS_DIR, "speedup_chart.png")

LABELS = {
    ("naive", "1"): "naive",
    ("cache_friendly", "1"): "cache-friendly",
    ("threaded", "2"): "threaded (2t)",
    ("threaded", "4"): "threaded (4t)",
    ("cuda_gpu", "1"): "GPU (CUDA)",
}

BLUE = "#2a78d6"
ORANGE = "#eb6834"
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


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = read_csv(CPU_CSV_PATH) + read_csv(GPU_CSV_PATH)

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
    colors = [ORANGE if label == "GPU (CUDA)" else BLUE for label in labels]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    bars = ax.bar(labels, times, color=colors, width=0.55, zorder=3)

    ax.set_yscale("log")
    ax.set_ylabel("time (ms, log scale)", color=SECONDARY_INK)
    ax.set_title("512x512 matmul: naive -> cache-friendly -> threaded -> GPU",
                 color=PRIMARY_INK, fontsize=13, pad=16)

    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", colors=SECONDARY_INK)
    ax.tick_params(axis="y", colors=MUTED)

    for bar, t, s in zip(bars, times, speedups):
        time_str = f"{t:.2f} ms" if t < 1 else f"{t:.1f} ms"
        label = time_str if s == 1.0 else f"{time_str} ({s:.0f}x)"
        ax.text(bar.get_x() + bar.get_width() / 2, t * 1.15, label, ha="center", va="bottom",
                color=PRIMARY_INK, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
