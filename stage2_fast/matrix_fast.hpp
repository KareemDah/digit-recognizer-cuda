#pragma once

#include "matrix.hpp"

// Same result as Matrix::multiply, but loops in i-k-j order instead of
// i-j-k, so the inner loop walks b and the output row contiguously instead
// of striding through a column of b.
Matrix multiply_cache_friendly(const Matrix& a, const Matrix& b);

// Cache-friendly matmul, with output rows split across num_threads threads.
Matrix multiply_threaded(const Matrix& a, const Matrix& b, unsigned num_threads);
