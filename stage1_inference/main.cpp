#include <iostream>

#include "matrix.hpp"
#include "mnist_loader.hpp"

namespace {

int predict(const MlpWeights& weights, const Matrix& image) {
    Matrix hidden = Matrix::multiply(image, weights.w1);
    hidden.add_in_place(weights.b1);
    hidden.relu_in_place();

    Matrix output = Matrix::multiply(hidden, weights.w2);
    output.add_in_place(weights.b2);
    output.softmax_row_in_place();

    return output.argmax_row();
}

}  // namespace

int main() {
    MlpWeights weights = load_weights("data/weights.bin");
    TestSet test_set = load_test_set("data/test_set.bin");

    int correct = 0;
    for (std::size_t i = 0; i < test_set.images.size(); ++i) {
        int predicted = predict(weights, test_set.images[i]);
        int actual = test_set.labels[i];
        if (predicted == actual) {
            ++correct;
        }
        if (i < 10) {
            std::cout << "image " << i << ": predicted=" << predicted
                      << " actual=" << actual << (predicted == actual ? "" : "  <-- wrong")
                      << "\n";
        }
    }

    double accuracy = static_cast<double>(correct) / static_cast<double>(test_set.images.size());
    std::cout << "\naccuracy: " << correct << "/" << test_set.images.size() << " = " << accuracy
              << "\n";

    return 0;
}
