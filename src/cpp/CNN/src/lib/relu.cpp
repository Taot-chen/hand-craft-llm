#include "relu.h"

void ReluLayer::forward(tensor_t<float>& in) {
    this->in = in;
    for (int i = 0; i < in.size.x; i++) {
        for (int j = 0; j < in.size.y; j++) {
            for (int k = 0; k < in.size.z; k++) {
                this->out(i, j, k) = in(i, j, k) < 0 ? 0 : in(i, j, k);
            }
        }
    }
}

void ReluLayer::backward(tensor_t<float>& grad_next_layer) {
    for (int i = 0; i < this->in.size.x; i++) {
        for (int j = 0; j < this->in.size.y; j++) {
            for (int k = 0; k < this->in.size.z; k++) {
                this->grad_in(i, j, k) = this->in(i, j, k) < 0 ? 0 : grad_next_layer(i, j, k);
            }
        }
    }
}
