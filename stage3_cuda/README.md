# Stage 3 — CUDA port

`matmul.cu` is a naive CUDA matmul kernel: one GPU thread computes one output
cell. It benchmarks itself against a CPU reference and checks correctness
before reporting any timing (same principle as Stage 2's benchmark).

This machine has no NVIDIA GPU, so this is compiled and run on a free cloud
GPU instead — Google Colab's free tier includes a T4 GPU and has `nvcc`
preinstalled, no setup required.

## Running it on Google Colab

1. Go to <https://colab.research.google.com/> and create a new notebook.
2. **Runtime → Change runtime type → Hardware accelerator → GPU (T4)**, save.
3. In the first cell, paste the *entire contents* of `matmul.cu` into a
   `%%writefile` cell so it gets saved as a file inside the Colab VM:

   ```
   %%writefile matmul.cu
   <paste the whole file here>
   ```

   Run that cell (Shift+Enter).
4. In a new cell, compile and run it:

   ```
   !nvcc -O3 matmul.cu -o matmul && ./matmul
   ```

5. Paste the output back here (or just tell me what it printed) — I'll help
   read the results and, if `nvcc` reports any compile errors, fix them on
   the spot. That's the normal workflow for developing CUDA code without a
   local GPU: write it, run it on real hardware, iterate on whatever the
   compiler/runtime actually says.

## Expected output shape

```
correctness check -- max abs diff vs CPU reference: 0.000000
correctness OK

matrix size: 512x512
GPU (naive CUDA kernel): <N> ms
CPU (naive, single-threaded): <N> ms
speedup: <N>x
```

Once you have real numbers, they get added to `results/benchmarks.csv` and
the chart, same as Stage 2.
