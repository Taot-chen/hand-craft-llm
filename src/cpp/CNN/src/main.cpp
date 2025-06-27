#include <bits/stdc++.h>
#include "lib/modeling_cnn.h"
#include "utils.h"
#include "logger.h"
#include <chrono>

int main() {
    std::vector<case_t> train_cases = read_train_cases();

    ModelCNN model;
    // TODO: change the hard parameters to more general code
    model.conv_layer(1, 5, 8, {28, 28, 1});
    model.relu_layer(model.output_size());
    model.pool_layer(2, 2, model.output_size());

    model.conv_layer(1, 3, 10, model.output_size());
    model.relu_layer(model.output_size());
    model.pool_layer(2, 2, model.output_size());

    model.fc_layer(model.output_size(), 10);

    float total_err = 0;
    int cnt = 0;
    float acc = 0;
    PRINT_CYAN("Start Training...");
    int epoches = 2;
    while (epoches--) {
        auto start1 = std::chrono::high_resolution_clock::now();
        for (case_t& c: train_cases) {
            float err = model.train(c.data, c.out);
            total_err += err;
            cnt++;
            acc += c.out(model.inference(), 0, 0) > 0.5 ? 1.0f : 0.0f;
            if (cnt %1000 == 0) {
                auto dura1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start1);
                start1 = std::chrono::high_resolution_clock::now();
                std::cout << "cases: " << cnt << " err=" << total_err / cnt
                    << " acc=" << acc / 1000.0f
                    << " " << 1000.0f / dura1.count() * 1000 << " pics/s"
                    << std::endl;
                acc = 0;
            }
        }
    }
    PRINT_CYAN("Start Testing...");
    std::vector<case_t>test_cases = read_test_cases();
    acc = total_err = 0;
    for (case_t& c: test_cases) {
        model.forward(c.data);
        acc += c.out(model.inference(), 0, 0) > 0.5 ? 1.0f : 0.0f;
    }
    std::cout << "Testing results: acc=" << acc / (float)(test_cases.size()) << std::endl;

    return 0;
}
