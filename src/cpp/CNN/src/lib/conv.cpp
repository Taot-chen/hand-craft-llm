#include "conv.h"

ConvLayer::ConvLayer(uint16_t _stride, uint16_t _kernel_size, uint16_t _num_kernel, td_size in_size):
    stride(_stride),
    kernel_size(_kernel_size),
    layer_t(layer_type::conv, in_size, {
        (in_size.x - _kernel_size) / _stride + 1,
        (in_size.y - _kernel_size) / _stride + 1,
        _num_kernel
    }) {
        // initialize kernels
        for (int n = 0; n < _num_kernel; n++) {
            tensor_t<float> new_w(_kernel_size, _kernel_size, in_size.z);
            int N = _kernel_size * _kernel_size * in_size.z;
            // initialize new_w
            for (int i = 0; i < _kernel_size; i++) {
                for (int j = 0; j < _kernel_size; j++) {
                    for (int k = 0; k < in_size.z; k++) {
                        new_w(i, j, k) = 1.0f / N * rand() / 2147483647.0;
                    }
                }
            }
            this->kernels.push_back(new_w);
        }
        for (int i = 0; i < _num_kernel; i++) {
            tensor_t<gradient_t> t(_kernel_size, _kernel_size, in_size.z);
            this->kernel_grads.push_back(t);
        }
    }


int ConvLayer::get_r (float f, int max_v, bool lim_min) {
    if (f <= 0) return  0;
    max_v -= 1;
    if (f >= max_v) return max_v;
    if (lim_min) return ceil(f);
    else return floor(f);
}


ConvLayer::range_t ConvLayer::map_to_output(int x, int y) {
    float a = x, b = y;
    return {
        this->get_r(
            (a - this->kernel_size + 1) / this->stride,
            this->out.size.x,
            true
        ),
        this->get_r(
            (b - this->kernel_size + 1) / this->stride,
            this->out.size.y,
            true
        ),
        0,
        this->get_r(
            a / this->stride,
            this->out.size.x,
            false
        ),
        this->get_r(
            b / this->stride,
            this->out.size.y,
            false
        ),
        (int)out.size.z - 1
    };
}


void ConvLayer::forward (tensor_t<float>& in) {
    this->in = in;
    for (int n = 0; n < this->kernels.size(); n++) {
        tensor_t<float>& kernel = this->kernels[n];
        for (int i = 0; i < in.size.x; i++) {
            for (int j = 0; j < in.size.y; j++) {
                td_size mapped = this->map_to_input({(uint16_t)i, (uint16_t)j, 0}, 0);
                float total = 0;
                for (int ini = 0; ini < this->kernel_size; ini++) {
                    for (int inj = 0; inj < this->kernel_size; inj++) {
                        for (int k = 0; k < in.size.z; k++) {
                            total += kernel(ini, inj, k) * in(mapped.x + ini, mapped.y + inj, k);
                        }
                    }
                }
                this->out(i, j, n) = total;
            }
        }
    }
}


void ConvLayer::update_weights () {
    for (int n = 0; n < this->kernels.size(); n++) {
        for (int i = 0; i < this->kernel_size; i++) {
            for (int j = 0; j < this->kernel_size; j++) {
                for (int k = 0; k < this->in.size.z; k++) {
                    float& w = this->kernels[n].get(i, j, k);
                    gradient_t& grad = this->kernel_grads[n].get(i, j, k);
                    w = update_weight(w, grad);
                    update_gradient(grad);
                }
            }
        }
    }
}


void ConvLayer::backward (tensor_t<float>& grad_next_layer) {
    for (int n = 0; n < this->kernels.size(); n++) {
        for (int i = 0; i < this->kernel_size; i++) {
            for (int j = 0; j < this->kernel_size; j++) {
                for (int k = 0; k < this->in.size.z; k++) {
                    this->kernel_grads[n](i, j, k).grad = 0;
                }
            }
        }
    }

    for (int i = 0; i < this->in.size.x; i++) {
        for (int j = 0; j < this->in.size.y; j++) {
            ConvLayer::range_t rn = this->map_to_output(i, j);
            for (int k = 0; k < this->in.size.z; k++) {
                float total_err = 0;
                // out[i, j, k] -> in[x, y, z] 有贡献的位置
                for (int ini = rn.min_x; ini <= rn.max_x; ini++) {
                    int min_x = ini * this->stride;
                    for (int  inj = rn.min_y; inj <= rn.max_y; inj++) {
                        int min_y = inj * this->stride;
                        for (int ink = rn.min_z; ink <= rn.max_z; ink++) {
                            // 贡献的系数 -> 第 k 个核作用 out[ i, j, k] 对应的 in 区域，in[x, y, z] 的系数
                            int kk = this->kernels[ink](i - min_x, j - min_y, k);
                            // 系数 * 偏导
                            total_err += kk * grad_next_layer(ini, inj, ink);
                            // kernel grad 同理
                            this->kernel_grads[ink](i - min_x, j - min_y, k).grad += this->in(i, j, k) * grad_next_layer(ini, inj, ink);
                        }
                    }
                }
                this->grad_in(i, j, k) = total_err;
            }
        }
    }
}
