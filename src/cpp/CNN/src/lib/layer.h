#pragma once
#include "tensor.h"
#include "gradient.h"

enum class layer_type {
    conv,
    fc,
    relu,
    pool,
    dropout
};

class layer_t {
    public:
        layer_type _type;
        tensor_t<float> grad_in;
        tensor_t<float> in;
        tensor_t<float> out;
        layer_t (layer_t _type_, td_size in_size, td_size out_size):
            type(_type_),
            in(in_size.x, in_size.y, in_size.z),
            out(out_size.x, out_size.y, out_size.z),
            grad_in(in_size.x, in_size.y, in_size.z)
        {}
        virtual ~layer(){}
        virtual void forward(tensor_t<float>& in) = 0;
        virtual void backward(tensor_t<float>& grad_next_layer) = 0;
        virtual void update_weights() = 0;
};
