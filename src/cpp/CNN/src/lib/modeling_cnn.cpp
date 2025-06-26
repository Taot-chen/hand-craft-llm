#pragma once
#include "modeling_cnn.h"

void ModelCNN::conv_layer (uint16_t stride, uint16_t kernel_size, uint16_t num_kernel, td_size in_size) {
    ConvLayer* layer = new ConvLayer(stride, kernel_size, num_kernel, in_size);
    this->layers.push_back((layer_t*)layer);
}


void ModelCNN::relu_layer (td_size in_size) {
    ReluLayer* layer = new ReluLayer(in_size);
    this->layers.push_back((layer_t*)layer);
}


void ModelCNN::pool_layer (uint16_t stride, uint16_t kernel_size, td_size in_size) {
    PoolLayer* layer = new PoolLayer(stride, kernel_size, in_size);
    this->layers.push_back((layer_t*)layer);
}


void ModelCNN::fc_layer (td_size in_size, int out_size) {
    FcLayer* layer = new FcLayer(in_size, out_size);
    this->layers.push_back((layer_t*)layer);
}


int ModelCNN::inference () {
    int ret = 0;
    // TODO: change the hard parameters to more general code
    for (int i = 0; i < 10; i++) {
        if (this->layers.back()->out(i, 0, 0) > this->layers.back()->out(ret, 0, 0)) ret  = i;
    }
    return ret;
}


void ModelCNN::forward (tensor_t<float>& input) {
    for (int i = 0; i < this->layers.size(); i++) {
        this->layers[i]->forward(i ? this->layers[i - 1]->out : input);
    }
}


float ModelCNN::train (tensor_t<float>& input, tensor_t<float>& label) {
    // forward
    this->forward(input);
    auto res_info = this->layers.back()->out() - label;

    // backward
    for (int i = this->layers.size() - 1; i >= 0; i--)
        this->layers[i]->backward(i < this->layers.size() - 1 ? this->layers[i + 1]->grad_in : res_info);

    // update weights
    for (int i = 0; i < this->layers.size(); i++)
        this->layers[i]->update_weights();

    float err = 0;
    for (int i = 0; i < 10; i++) {
        float res = label(i, 0, 0) - res_info(i, 0, 0);
        err += res * res;
    }
    // TODO: change the hard parameters to more general code
    return sqrt(err) * 100;
}
