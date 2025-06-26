#pragma once
#include "layer.h"

class DropoutLayer: public layer_t {
    public:
        tensor_t<bool> hitmap;
        float p_activation;

        DropoutLayer(td_size in_size, float p_activation):
            layer_t(layer_type::dropout, in_size, in_size),
            hitmap(in_size.x, in_size.y, in_size.z),
            p_activation(p_activation)
        {}
        void forward (tensor_t<float>& in) override;
        void update_weights() override {};
        void backward (tensor_t<float>& grad_next_layer) override;
};
