#pragma once
#include "layer.h"

class ConvLayer: public layer_t {
    public:
        std::vector<tensor_t<float>> kernels;
        std::vector<tensor_t<gradient_t>> kernel_grads;
        uint16_t stride;
        uint16_t kernel_size;

        ConvLayer(uint16_t stride, uint16_t kernel_size, uint16_t num_kernel, td_size in_size);
        td_size map_to_input(td_size out, int z) return {out.x * this->stride, out.y * stride, z}

        struct range_t {
            int min_x, min_y, min_z;
            int max_x, max_y, max_z;
        };

        int get_r (float f, int max_v, int lim_min);
        range_t map_to_output(int x, int y);
        void forward (tensor_t<float>& in) override;
        void update_weights () override;
        void backward (tensor_t<float>& grad_next_layer) override;
};
