#pragma once
#include "tensor.h"
#include "gradient.h"
#include "conv.h"
#include "dropout.h"
#include "fc.h"
#include "pool.h"
#include "relu.h"

class ModelCNN {
    private:
        std::vector<layer_t*> layers;
    public:
        ModelCNN () {}

        void conv_layer (uint16_t stride, uint16_t kernel_size, uint16_t num_kernel, td_size in_size);
        void relu_layer (td_size in_size);
        void pool_layer (uint16_t stride, uint16_t kernel_size, td_size in_size);
        void fc_layer (td_size in_size, int out_size);

        td_size& output_size() return this->layers.back()->out.size;

        int inference ();
        tensor_t<float>& infer_info() return this->layers.back()->out;
        void forward (tensor_t<float>& input);
        float train (tensor_t<float>& input, tensor_t<float>& label);
};