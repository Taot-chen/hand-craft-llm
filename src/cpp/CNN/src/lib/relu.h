#pragma once
#include "layer.h"

class ReluLayer: public layer_t {
    public:
        ReluLayer(td_size in_size): layer_t(layer_type::relu, in_size, in_size) {}
        void forward(tensor_t<float>& in) override;
        void update_weights() override {}
        void backward(tensor_t<float>& grad_next_layer) override;
}