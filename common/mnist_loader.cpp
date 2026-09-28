#include "mnist_loader.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

using namespace std;

namespace {

ifstream open_binary(const string& path) {
    ifstream file(path, ios::binary);
    if (!file) {
        throw runtime_error("could not open file: " + path);
    }
    return file;
}

uint32_t read_u32(ifstream& file) {
    uint32_t value = 0;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

}  // namespace

MlpWeights load_weights(const std::string& path) {
    ifstream file = open_binary(path);

    uint32_t input_dim = read_u32(file);
    uint32_t hidden_dim = read_u32(file);
    uint32_t output_dim = read_u32(file);

    MlpWeights weights{
        Matrix(input_dim, hidden_dim),
        Matrix(1, hidden_dim),
        Matrix(hidden_dim, output_dim),
        Matrix(1, output_dim),
    };

    file.read(reinterpret_cast<char*>(weights.w1.data()), input_dim * hidden_dim * sizeof(float));
    file.read(reinterpret_cast<char*>(weights.b1.data()), hidden_dim * sizeof(float));
    file.read(reinterpret_cast<char*>(weights.w2.data()), hidden_dim * output_dim * sizeof(float));
    file.read(reinterpret_cast<char*>(weights.b2.data()), output_dim * sizeof(float));

    return weights;
}

TestSet load_test_set(const std::string& path) {
    ifstream file = open_binary(path);

    uint32_t count = read_u32(file);
    uint32_t image_dim = read_u32(file);

    TestSet test_set;
    test_set.images.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        test_set.images.emplace_back(1, image_dim);
    }
    for (uint32_t i = 0; i < count; ++i) {
        file.read(reinterpret_cast<char*>(test_set.images[i].data()), image_dim * sizeof(float));
    }

    test_set.labels.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        test_set.labels[i] = static_cast<int>(read_u32(file));
    }

    return test_set;
}
