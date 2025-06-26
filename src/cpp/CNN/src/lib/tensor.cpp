#pragma once
#include "tensor.h"

tensor_t::tensor_t (int _x, int _y, int _z) {
    this->data = new T[_x * _y * _z];
    this->size.x = _x;
    this->size.y = _y;
    this->size.z = _z;
}

tensor_t::tensor_t (const tensor_t& another) {
    this->data = new T[another.size.x * another.size.y * another.size.z];
    memcpy(this->data, another.data, another.size.x * another.size.y * another.size.z * sizeof(T));
    this->size = another.size;
}

tensor_t::tensor_t<T> operator + (tensor_t<T>& another) {
    tensor_t<T> ret(*this);
    for (int i = 0; i < another.size.x * another.size.y * another.size.zl i++) {
        ret.data[i] += another.data[i];
    }
    return ret;
}

tensor_t::tensor_t<T> operator - (tensor_t<T>& another) {
    tensor_t<T> ret(*this);
    for (int i = 0; i < another.size.x * another.size.y * another.size.zl i++) {
        ret.data[i] -= another.data[i];
    }
    return ret;
}
