#pragma once
#include "layer.h"
#include <cfloat>

// MaxPool
class PoolLayer: public layer_t {
    public:
        uint16_t stride;
        uint16_t kernel_size;

        PoolLayer(uint16_t _stride, uint16_t _kernel_size, td_size in_size):
            stride (_stride),
            kernel_size(_kernel_size),
            layer_t(
                layer_type::pool,
                in_size,
                {(in_size.x - _kernel_size) / _stride + 1, (in_size.y - _kernel_size) / _stride + 1, in_size.z}
            )
        {}

        td_size map_to_input (td_size out, int z) {
            return {out.x * this->stride, out.y * this->stride, z};
        }

        struct range_t {
            int min_x, min_y, min_z;
            int max_x, max_y, max_z;
        };

        int get_r (float f, int max_v, bool lim_min);
        range_t map_to_output (int x, int y);
        void forward (tensor_t<float>& in) override;
        void update_weights () override {}
        void backward (tensor_t<float>& grad_next_layer) override;
};
