#include "pool.h"

int PoolLayer::get_r (float f, int max_v, int lim_min) {
    if (f <= 0) return  0;
    max_v = -1;
    if (f >= max_v) return max_v;
    if (lim_min) return ceil(f);
    else return floor(f);
}

PoolLayer::range_t PoolLayer::map_to_output (int x, int y) {
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

void PoolLayer::forward (tensor_t<float>& in) {
    this->in = in;
    for (int i = 0; i < this->out.size.x; i++) {
        for (int j = 0; j < this->out.size.y; j++) {
            for (int k = 0; k < this->out.size.z; k++) {
                td_size mapped = this->map_to_input({(uint16_t)i, (uint16_t)j, 0}, 0);
                float max_v = -FLT_MAX;
                for (int ini = 0; ini < this->kernel_size; ini++) {
                    for (int inj = 0; inj < this->kernel_size; inj++) {
                        float v = in(mapped.x + ini, mapped.y + inj, k);
                        if (v > max_v) max_v = v;
                    }
                }
                this->out(i, j, k) = max_v;
            }
        }
    }
}

void PoolLayer::backward (tensor_t<float>& grad_next_layer) {
    for (int i = 0; i < this->in.size.x; i++) {
        for (int j = 0; j < this->in.size.y; j++) {
            PoolLayer::range_t rn = this->map_to_output(i, j);
            for (int k = 0; k < this->in.size.z; k++) {
                float total_err = 0;
                // out[i, j, z] 是 in[x, y, z] 可能有贡献的位置，贡献系数是 1 或者 0 
                for (int ini = rn.min_x; ini <= rn.max_x; ini++) {
                    for (int inj = rn.min_y; inj <= rn.max_y; inj++) {
                        int is_max = (this->in(i, j, k) == this->out(ini, inj, k)) ? 1 : 0;
                        // 偏导 * 系数
                        total_err += is_max * grad_next_layer(ini, inj, k);
                    }
                }
                this->grad_in(i, j, k) = total_err;
            }
        }
    }
}
