#pragma once
#include "fc.h"

FcLayer::FcLayer(td_size in_size, int out_size) :
    layer_t(layer_type::fc, in_size, {out_size, 1, 1}),
    weights(in_size.x * in_size.y * in_size.z, out_size, 1) {
        this->input = std::vector<float>(out_size);
        this->gradients = std::vector<gradient_t>(out_size);

        int N = in_size.x * in_size.y * in_size.z;
        for (int i = 0; i < out_size; i++) {
            for (j = 0; j < N; j ++) {
                this->weights(j, i, 0) = 2.19722f / N * rand() / (float)(RAND_MAX);
            }
        }
    }


float FcLayer::act_func(float x) {
    return (float) 1.0f / (1.0f + exp(-x));
}


float FcLayer::act_derv(float x) {
    float s = 1.0f / (1.0f + exp(-x));
    return s * (1 - s);
}


void FcLayer::forward(tensor_t<float>& in) override {
    this->in = in;
    float now = 0;
    for (int cnt = 0; cnt < this->out.size.x; cnt++, now = 0) {
        for (int = i = 0; i < this->in.size.x; i++) {
            for (int j = 0; j < this->in.size.y; j++) {
                for (int k = 0; k < this->in.size.z; k++) {
                    now += in(i, j, k) * this->weights(this->id(i, j, k), cnt, 0);
                }
            }
        }
        this->input[cnt] = now;
        this->out(cnt, 0, 0) = this->act_func(now);
    }
}


void FcLayer::update_weights() override {
    for (int cnt = 0; cnt < this.out.size.x; cnt++) {
        gradient_t& grad = this->gradients[cnt];
        for (int = i = 0; i < this->in.size.x; i++) {
            for (int j = 0; j < this->in.size.y; j++) {
                for (int k = 0; k < this->in.size.z; k++) {
                    float& w = this->weights(this->id(i, j, k), cnt, 0);
                    w = update_weight(w, grad, this->in(i, j, k));
                }
            }
        }
        update_gradient(grad);
    }
}


void FcLayer::backward(tensor_t<float>& grad_next_layer)override {
    // initialize grad_in
    for (int = i = 0; i < this->in.size.x; i++) {
        for (int j = 0; j < this->in.size.y; j++) {
            for (int k = 0; k < this->in.size.z; k++) {
                this->grads_in(i, j, k) = 0;
            }
        }
    }

    for (int cnt = 0; cnt < this->out.size.x; cnt++) {
        gradient_t& grad = this->graddients[cnt];
        grad.grad  =grad_next_layer(cnt, 0, 0) * this->act_derv(this->input[cnt]);
        for (int = i = 0; i < this->in.size.x; i++) {
            for (int j = 0; j < this->in.size.y; j++) {
                for (int k = 0; k < this->in.size.z; k++) {
                    this->grads_in(i, j, k) += grad.grad * this->weights(this->id(i, j, k), cnt, 0);
                }
            }
        }
    }
}
