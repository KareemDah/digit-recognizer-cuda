# Runs the CPU benchmark and regenerates the chart in one step, so the
# chart never goes stale just because the second command got skipped.
# The GPU bar can't be refreshed this way -- there's no local GPU, so that
# number only ever comes from manually running stage3_cuda/matmul.cu on
# Colab and updating results/gpu_benchmarks.csv by hand.

$ErrorActionPreference = "Stop"

Write-Host "Building..."
cmake --build build

Write-Host "`nRunning CPU benchmark..."
& .\build\stage2_fast\stage2_benchmark.exe

Write-Host "`nRedrawing chart..."
python tools\make_chart.py

Write-Host "`nDone -- open results\speedup_chart.png to see it."
