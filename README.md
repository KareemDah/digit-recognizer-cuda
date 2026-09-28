# Digit Recognizer → CUDA

A from-scratch C++ progression: a CPU MNIST digit-recognition inference engine,
optimized with cache-friendly access and multithreading, then ported to CUDA
and benchmarked CPU vs. GPU. Built stage by stage while learning C++ along the way.

## Stages

- [ ] **Stage 1 — CPU inference engine**: matrix class + forward pass over a
      pretrained MNIST classifier.
- [ ] **Stage 2 — Make it fast**: cache-friendly matmul + multithreading, benchmarked.
- [ ] **Stage 3 — CUDA port**: matmul kernel on GPU, CPU vs. GPU timings
      (validated on a cloud GPU — see `stage3_cuda/README.md`).
- [ ] **Stage 4 — Stretch: tiny language model** (llama2.c-style), only if time allows.

## Building

Requires CMake and a C++17 compiler (developed with MinGW-w64 g++ on Windows).

```
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
```

## Data / weights

`tools/train_export.py` trains a tiny MLP on MNIST with plain numpy and exports
raw weights + a handful of test images for the C++ engine to load:

```
pip install numpy
python tools/train_export.py
```

## Benchmarks

See `results/` (filled in during Stage 2/3).
