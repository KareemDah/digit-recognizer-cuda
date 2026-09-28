"""
Trains a tiny 784-128-10 MLP on MNIST using plain numpy (no torch), then
exports the weights and a handful of test images as raw binary files that
the C++ inference engine (stage1_inference) loads directly.

This script is a one-time data-prep tool, not part of the C++ project itself.
Run it once: `pip install numpy` then `python tools/train_export.py`.
"""

import gzip
import os
import struct
import urllib.request

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

INPUT_DIM = 28 * 28
HIDDEN_DIM = 128
OUTPUT_DIM = 10
NUM_TEST_EXPORT = 200  # how many test images to export for the C++ side


def download_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    for key, filename in MNIST_FILES.items():
        dest = os.path.join(DATA_DIR, filename)
        if os.path.exists(dest):
            continue
        url = MNIST_BASE_URL + filename
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)


def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic number for images: {magic}"
        buf = f.read(num * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows * cols)
        return data.astype(np.float32) / 255.0


def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic number for labels: {magic}"
        buf = f.read(num)
        return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(x_train, y_train, x_test, y_test, epochs=8, batch_size=128, lr=0.1):
    rng = np.random.default_rng(0)
    w1 = rng.normal(0, np.sqrt(2.0 / INPUT_DIM), (INPUT_DIM, HIDDEN_DIM)).astype(np.float32)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    w2 = rng.normal(0, np.sqrt(2.0 / HIDDEN_DIM), (HIDDEN_DIM, OUTPUT_DIM)).astype(np.float32)
    b2 = np.zeros(OUTPUT_DIM, dtype=np.float32)

    y_onehot = np.eye(OUTPUT_DIM, dtype=np.float32)[y_train]
    n = x_train.shape[0]

    for epoch in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = x_train[idx]
            yb = y_onehot[idx]
            b = xb.shape[0]

            # forward
            z1 = xb @ w1 + b1
            a1 = np.maximum(z1, 0)  # ReLU
            z2 = a1 @ w2 + b2
            probs = softmax(z2)

            # backward (cross-entropy + softmax gradient)
            dz2 = (probs - yb) / b
            dw2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)

            da1 = dz2 @ w2.T
            dz1 = da1 * (z1 > 0)
            dw1 = xb.T @ dz1
            db1 = dz1.sum(axis=0)

            w1 -= lr * dw1
            b1 -= lr * db1
            w2 -= lr * dw2
            b2 -= lr * db2

        acc = evaluate(x_test, y_test, w1, b1, w2, b2)
        print(f"epoch {epoch + 1}/{epochs}  test accuracy: {acc:.4f}")

    return w1, b1, w2, b2


def evaluate(x, y, w1, b1, w2, b2):
    z1 = x @ w1 + b1
    a1 = np.maximum(z1, 0)
    z2 = a1 @ w2 + b2
    preds = np.argmax(z2, axis=1)
    return float((preds == y).mean())


def export_weights(path, w1, b1, w2, b2):
    with open(path, "wb") as f:
        f.write(struct.pack("<III", INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM))
        f.write(w1.astype("<f4").tobytes())
        f.write(b1.astype("<f4").tobytes())
        f.write(w2.astype("<f4").tobytes())
        f.write(b2.astype("<f4").tobytes())


def export_test_set(path, x, y, count):
    x = x[:count]
    y = y[:count]
    with open(path, "wb") as f:
        f.write(struct.pack("<II", count, INPUT_DIM))
        f.write(x.astype("<f4").tobytes())
        f.write(y.astype("<u4").tobytes())


def main():
    download_mnist()

    x_train = read_idx_images(os.path.join(DATA_DIR, MNIST_FILES["train_images"]))
    y_train = read_idx_labels(os.path.join(DATA_DIR, MNIST_FILES["train_labels"]))
    x_test = read_idx_images(os.path.join(DATA_DIR, MNIST_FILES["test_images"]))
    y_test = read_idx_labels(os.path.join(DATA_DIR, MNIST_FILES["test_labels"]))

    print(f"train: {x_train.shape}, test: {x_test.shape}")

    w1, b1, w2, b2 = train(x_train, y_train, x_test, y_test)

    final_acc = evaluate(x_test, y_test, w1, b1, w2, b2)
    print(f"final numpy reference accuracy: {final_acc:.4f}")

    weights_path = os.path.join(DATA_DIR, "weights.bin")
    test_path = os.path.join(DATA_DIR, "test_set.bin")
    export_weights(weights_path, w1, b1, w2, b2)
    export_test_set(test_path, x_test, y_test, NUM_TEST_EXPORT)

    print(f"wrote {weights_path}")
    print(f"wrote {test_path} ({NUM_TEST_EXPORT} test images)")


if __name__ == "__main__":
    main()
