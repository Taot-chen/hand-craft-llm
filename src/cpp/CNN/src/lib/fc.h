#pragma once
#include <cmath>
#include <cfloat>
#include <cstring>
#include "layer.h"

class FcLayer: public layer_t {
    public:
        std::vector<float> input;
        tensor_t<float> weights;
        std::vector<gradient_t> gradients;

        FcLayer(td_size in_size, int out_size);

        // 铺平后的id
        int id (int x, int y, int z) {
            return z * (this->in.size.x * this->in.size.y) + y * (this->in.size.x) + x;
        }

        // activation, sigmoid
        float act_func(float x);

        // activation derivative
        float act_derv(float x);

        void forward(tensor_t<float>& in) override;
        void update_weights() override;
        void backward(tensor_t<float>& grad_next_layer)override;
};
