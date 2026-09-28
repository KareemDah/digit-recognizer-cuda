// Naive CUDA matmul: one GPU thread computes one output cell.
// Mirrors common/matrix.cpp's Matrix::multiply (same math, same naive
// algorithm) so the CPU and GPU versions can be checked against each other.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace std;

// __global__ marks this as a "kernel": a function that runs on the GPU
// (device), launched from CPU (host) code, executed once per thread.
__global__ void matmul_kernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // n isn't always an exact multiple of the block size, so some threads
    // at the edges don't correspond to a real output cell -- skip them.
    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

// Same naive matmul as common/matrix.cpp's Matrix::multiply, but running
// here on the CPU host so we have a trusted reference to check the GPU
// result against -- same principle as Stage 2's correctness check.
void matmul_cpu_reference(const vector<float>& a, const vector<float>& b, vector<float>& c,
                           int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err = (call);                                                   \
        if (err != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(err));                                       \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

int main() {
    const int n = 512;
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    // Host-side (CPU) data -- same random matrices as Stage 2's benchmark.
    vector<float> h_a(n * n), h_b(n * n), h_c(n * n), h_reference(n * n);
    mt19937 rng(42);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n * n; ++i) {
        h_a[i] = dist(rng);
        h_b[i] = dist(rng);
    }

    // Device-side (GPU) memory. This is separate physical memory from the
    // host's RAM -- the CPU cannot read/write it directly, only through
    // cudaMemcpy.
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Copy the input matrices from host memory to device memory.
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Each block is a 16x16 team of threads (256 threads, a common choice).
    // The grid is sized so there's enough blocks to cover all n x n cells,
    // rounding up in case n isn't a multiple of 16.
    dim3 block_size(16, 16);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                    (n + block_size.y - 1) / block_size.y);

    // Warm-up launch (unmeasured): the very first kernel launch on a GPU
    // often includes a one-time cost -- the driver JIT-compiling the kernel
    // for this exact GPU's architecture -- that has nothing to do with the
    // algorithm itself. Run it once and throw the result away before timing
    // anything, same reasoning as Stage 2's best-of-N: don't let a one-off
    // hiccup pollute the measurement.
    matmul_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // cudaEvent-based timing: GPU work is asynchronous (the kernel launch
    // below returns immediately on the CPU side), so a CPU chrono clock
    // would measure almost nothing. cudaEvents are timestamps recorded on
    // the GPU's own timeline instead.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int repeats = 5;
    float gpu_ms = -1.0f;
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        matmul_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (gpu_ms < 0.0f || ms < gpu_ms) {
            gpu_ms = ms;
        }
    }

    // Copy the result back from device memory to host memory.
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Correctness check against a trusted CPU reference, same principle as
    // Stage 2: don't trust a speed number until the result is verified.
    matmul_cpu_reference(h_a, h_b, h_reference, n);
    float max_abs_diff = 0.0f;
    for (int i = 0; i < n * n; ++i) {
        max_abs_diff = max(max_abs_diff, fabs(h_c[i] - h_reference[i]));
    }
    printf("correctness check -- max abs diff vs CPU reference: %f\n", max_abs_diff);
    if (max_abs_diff > 1e-2f) {
        fprintf(stderr, "FAILED: GPU result does not match CPU reference\n");
        return 1;
    }
    printf("correctness OK\n\n");

    printf("matrix size: %dx%d\n", n, n);
    printf("GPU (naive CUDA kernel): %f ms\n", gpu_ms);

    // Also time the CPU reference the same way Stage 2 did (best-of-N), for
    // a direct CPU-vs-GPU comparison in this one program.
    double cpu_ms = -1.0;
    for (int i = 0; i < repeats; ++i) {
        auto cpu_start = chrono::steady_clock::now();
        matmul_cpu_reference(h_a, h_b, h_reference, n);
        auto cpu_end = chrono::steady_clock::now();
        double ms = chrono::duration<double, milli>(cpu_end - cpu_start).count();
        if (cpu_ms < 0.0 || ms < cpu_ms) {
            cpu_ms = ms;
        }
    }
    printf("CPU (naive, single-threaded): %f ms\n", cpu_ms);
    printf("speedup: %fx\n", cpu_ms / gpu_ms);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
