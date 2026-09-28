#pragma once

#include <string>
#include <vector>

#include "matrix.hpp"

struct MlpWeights {
    Matrix w1;  // input_dim x hidden_dim
    Matrix b1;  // 1 x hidden_dim
    Matrix w2;  // hidden_dim x output_dim
    Matrix b2;  // 1 x output_dim
};

struct TestSet {
    std::vector<Matrix> images;  // each 1 x input_dim
    std::vector<int> labels;
};

MlpWeights load_weights(const std::string& path);
TestSet load_test_set(const std::string& path);
