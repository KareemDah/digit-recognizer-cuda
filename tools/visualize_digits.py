"""
Renders a handful of images from data/test_set.bin as an actual picture, so
you can see what the C++ program is looking at. One-off helper, not part of
the C++ project.

Run: python tools/visualize_digits.py
"""

import os
import struct

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

# predictions from a real run of stage1_inference.exe, so the picture and
# the program's actual output line up
KNOWN_PREDICTIONS = {
    0: 7,
    1: 2,
    2: 1,
    3: 0,
    8: 6,  # this one the model got wrong (actual digit is 5)
}


def load_test_set(path):
    with open(path, "rb") as f:
        count, image_dim = struct.unpack("<II", f.read(8))
        images = np.frombuffer(f.read(count * image_dim * 4), dtype="<f4").reshape(
            count, image_dim
        )
        labels = np.frombuffer(f.read(count * 4), dtype="<u4")
        return images, labels


def main():
    images, labels = load_test_set(os.path.join(DATA_DIR, "test_set.bin"))

    indices = [0, 1, 2, 3, 8]
    fig, axes = plt.subplots(1, len(indices), figsize=(12, 3))

    for ax, idx in zip(axes, indices):
        img = images[idx].reshape(28, 28)
        actual = int(labels[idx])
        predicted = KNOWN_PREDICTIONS.get(idx)
        correct = predicted == actual

        ax.imshow(img, cmap="gray_r")
        ax.set_title(
            f"actual: {actual}\npredicted: {predicted}",
            color="#0b0b0b" if correct else "#d03b3b",
            fontsize=11,
        )
        ax.axis("off")

    fig.suptitle("What the C++ program actually sees and predicts", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "sample_digits.png")
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
