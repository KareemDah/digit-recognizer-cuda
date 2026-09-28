#include "matrix_fast.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace std;

namespace {

// Fills result's rows [row_begin, row_end) using the i-k-j loop order.
void multiply_rows(const Matrix& a, const Matrix& b, Matrix& result, size_t row_begin,
                    size_t row_end) {
    size_t a_cols = a.cols();
    size_t b_cols = b.cols();
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* result_data = result.data();

    for (size_t i = row_begin; i < row_end; ++i) {
        float* result_row = result_data + i * b_cols;
        const float* a_row = a_data + i * a_cols;
        for (size_t k = 0; k < a_cols; ++k) {
            float a_val = a_row[k];
            const float* b_row = b_data + k * b_cols;
            for (size_t j = 0; j < b_cols; ++j) {
                result_row[j] += a_val * b_row[j];
            }
        }
    }
}

}  // namespace

Matrix multiply_cache_friendly(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows()) {
        throw invalid_argument("multiply_cache_friendly: shape mismatch");
    }
    Matrix result(a.rows(), b.cols());
    multiply_rows(a, b, result, 0, a.rows());
    return result;
}

// Each worker thread only ever writes to its own slice of result's rows, so
// there's no overlap between threads and no lock is needed.
Matrix multiply_threaded(const Matrix& a, const Matrix& b, unsigned num_threads) {
    if (a.cols() != b.rows()) {
        throw invalid_argument("multiply_threaded: shape mismatch");
    }
    if (num_threads == 0) {
        num_threads = 1;
    }

    Matrix result(a.rows(), b.cols());
    size_t total_rows = a.rows();
    size_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;

    vector<thread> workers;
    for (unsigned t = 0; t < num_threads; ++t) {
        size_t row_begin = t * rows_per_thread;
        size_t row_end = min(row_begin + rows_per_thread, total_rows);
        if (row_begin >= row_end) {
            break;
        }
        workers.emplace_back(multiply_rows, cref(a), cref(b), ref(result), row_begin, row_end);
    }
    for (thread& w : workers) {
        w.join();
    }
    return result;
}
