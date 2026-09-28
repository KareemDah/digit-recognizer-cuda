#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "matrix.hpp"
#include "matrix_fast.hpp"

using namespace std;

namespace {

Matrix random_matrix(size_t rows, size_t cols, mt19937& rng) {
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Matrix m(rows, cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            m.at(r, c) = dist(rng);
        }
    }
    return m;
}

// Largest absolute difference between corresponding elements. The fast
// variants sum in a different order than the naive one, so tiny
// floating-point rounding differences are expected -- this just checks
// they're actually computing the same matrix, not a different algorithm.
float max_abs_diff(const Matrix& x, const Matrix& y) {
    float worst = 0.0f;
    for (size_t r = 0; r < x.rows(); ++r) {
        for (size_t c = 0; c < x.cols(); ++c) {
            worst = max(worst, fabs(x.at(r, c) - y.at(r, c)));
        }
    }
    return worst;
}

// Runs fn `repeats` times and returns the fastest time, in milliseconds.
// We take the best-of-N rather than an average because occasional OS
// scheduling hiccups inflate a run without telling us anything about the
// algorithm itself.
template <typename Fn>
double best_of_ms(Fn&& fn, int repeats) {
    double best = -1.0;
    for (int i = 0; i < repeats; ++i) {
        auto start = chrono::steady_clock::now();
        fn();
        auto end = chrono::steady_clock::now();
        double ms = chrono::duration<double, milli>(end - start).count();
        if (best < 0 || ms < best) {
            best = ms;
        }
    }
    return best;
}

}  // namespace

int main() {
    mt19937 rng(42);
    const size_t n = 512;
    Matrix a = random_matrix(n, n, rng);
    Matrix b = random_matrix(n, n, rng);

    const int repeats = 3;
    unsigned hw_threads = thread::hardware_concurrency();
    if (hw_threads == 0) {
        hw_threads = 4;
    }

    cout << "matrix size: " << n << "x" << n << ", hardware_concurrency: " << hw_threads
         << "\n\n";

    Matrix reference = Matrix::multiply(a, b);
    float diff_cache = max_abs_diff(reference, multiply_cache_friendly(a, b));
    float diff_threaded = max_abs_diff(reference, multiply_threaded(a, b, hw_threads));
    cout << "correctness check -- max abs diff vs naive: cache-friendly=" << diff_cache
         << ", threaded=" << diff_threaded << "\n";
    if (diff_cache > 1e-2f || diff_threaded > 1e-2f) {
        cerr << "FAILED: fast matmul variants do not match naive result\n";
        return 1;
    }
    cout << "correctness OK\n\n";

    // Written separately from results/gpu_benchmarks.csv (which the CUDA
    // run's number lives in) so that re-running this CPU benchmark can
    // never overwrite the GPU number -- they're two different programs on
    // two different machines, so they get two different files.
    ofstream csv("results/cpu_benchmarks.csv");
    csv << "variant,threads,size,time_ms\n";

    double naive_ms = best_of_ms([&] { Matrix r = Matrix::multiply(a, b); }, repeats);
    cout << "naive:               " << naive_ms << " ms\n";
    csv << "naive,1," << n << "," << naive_ms << "\n";

    double cache_ms = best_of_ms([&] { Matrix r = multiply_cache_friendly(a, b); }, repeats);
    cout << "cache-friendly (1t): " << cache_ms << " ms  (" << (naive_ms / cache_ms) << "x)\n";
    csv << "cache_friendly,1," << n << "," << cache_ms << "\n";

    vector<unsigned> thread_counts = {2, 4, hw_threads};
    for (unsigned t : thread_counts) {
        double ms = best_of_ms([&] { Matrix r = multiply_threaded(a, b, t); }, repeats);
        cout << "threaded (" << t << "t):        " << ms << " ms  (" << (naive_ms / ms)
             << "x)\n";
        csv << "threaded," << t << "," << n << "," << ms << "\n";
    }

    cout << "\nwrote results/cpu_benchmarks.csv\n";
    return 0;
}
