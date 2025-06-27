#include "dropout.h"

void DropoutLayer::forward(tensor_t<float>& in) {
    this->in = in;
    for (int i = 0; i < in.size.x * in.size.y * in.size.z; i++) {
        bool active = (rand() % RAND_MAX) / float(RAND_MAX) < this->p_activation;
        this->hitmap.data[i] = active;
        this->out.data[i] = active ? in.data[i] : 0.0f;
    }
}

void DropoutLayer::backward (tensor_t<float>& grad_next_layer) {
    for (int i = 0; i < this->in.size.x * this->in.size.y * this->in.size.z; i++) {
        this->grad_in.data[i] = this->hitmap.data[i] ? grad_next_layer.data[i] : 0.0f;
    }
}
