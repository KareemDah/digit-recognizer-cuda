#include "matrix.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

float& Matrix::at(std::size_t row, std::size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw out_of_range("Matrix::at index out of range");
    }
    return data_[row * cols_ + col];
}

float Matrix::at(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw out_of_range("Matrix::at index out of range");
    }
    return data_[row * cols_ + col];
}

// Naive triple-loop matmul (i-j-k order). Stage 2 revisits this loop order
// for cache-friendliness — this version is the "before" baseline.
Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
    if (a.cols_ != b.rows_) {
        throw invalid_argument("Matrix::multiply: shape mismatch");
    }
    Matrix result(a.rows_, b.cols_);
    for (std::size_t i = 0; i < a.rows_; ++i) {
        for (std::size_t j = 0; j < b.cols_; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < a.cols_; ++k) {
                sum += a.data_[i * a.cols_ + k] * b.data_[k * b.cols_ + j];
            }
            result.data_[i * result.cols_ + j] = sum;
        }
    }
    return result;
}

void Matrix::add_in_place(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw invalid_argument("Matrix::add_in_place: shape mismatch");
    }
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
}

void Matrix::relu_in_place() {
    for (float& v : data_) {
        v = max(0.0f, v);
    }
}

void Matrix::softmax_row_in_place() {
    if (rows_ != 1) {
        throw invalid_argument("Matrix::softmax_row_in_place: expected a single row");
    }
    float max_val = *max_element(data_.begin(), data_.end());
    float sum = 0.0f;
    for (float& v : data_) {
        v = exp(v - max_val);
        sum += v;
    }
    for (float& v : data_) {
        v /= sum;
    }
}

int Matrix::argmax_row() const {
    if (rows_ != 1) {
        throw invalid_argument("Matrix::argmax_row: expected a single row");
    }
    auto it = max_element(data_.begin(), data_.end());
    return static_cast<int>(distance(data_.begin(), it));
}
