#pragma once

#include <cstddef>
#include <vector>

class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols);

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    float& at(std::size_t row, std::size_t col);
    float at(std::size_t row, std::size_t col) const;

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    static Matrix multiply(const Matrix& a, const Matrix& b);

    void add_in_place(const Matrix& other);
    void relu_in_place();
    void softmax_row_in_place();
    int argmax_row() const;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<float> data_;
};
