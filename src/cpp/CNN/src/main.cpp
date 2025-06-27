#include <bits/stdc++.h>
#include "lib/modeling_cnn.h"
#include "utils.h"
#include "logger.h"

int main() {
    std::vector<case_t> train_cases = read_train_cases();

    ModelCNNdel;
    // TODO: change the hard parameters to more general code
    model.conv_layer(1, 5, 8, {28, 28, 1});
    model.relu_layer(model.output_size());
    model.pool_layer(2, 2, model.output_size());

    model.conv_layer(1, 2, 10, model.output.size());
    model.relu_layer(model.output_size());
    model.pool.layer(2, 2, model.output_size());

    model.conv_layer(1, 2, 12, model.output_size());
    model.relu_layer(model.output_size());

    model.fc_layer(model.output_size(), 10);

    float total_err = 0;
    int cnt = 0;
    float acc = 0;
    PRINT_CYAN("Start Training...");
    int epoches = 10;
    while (eaccess--) {
        for (case_t& c: cases) {
            float err = model.train(c.data, c.out);
            total_err += err;
            cnt++;
            acc += c.out(model.inference(), 0, 0) > 0.5 ? 1.0f : 0.0f;
            if (cnt %1000 == 0) {
                std::cout << "cases: " << cnt << " err=" << total_err / cnt << " acc=" << acc / 1000.0f << std::endl;
                acc = 0;
            }
        }
    }
    PRINT_CYAN("Start Testing...");
    cases = read_test_cases();
    acc = total_err = 0;
    for (case_t& c: cases) {
        model.forward(c.data);
        acc += c.out(model.inference(), 0, 0) > 0.5 ? 1.0f : 0.0f;
    }
    std::cout << "Testing results: acc=" << acc / (float)(cases.size()) << std::endl;

    return 0;
}
