#pragma once
#include <cassert>
#include <vector>
#include <cstring>

typedef struct dims_t {
    int x, y, z;
} td_size;

template<typename T>
class tensor_t {
    private:
        T* data;
        td_size size;
    public:
        tensor_t (int _x, int _y, int _z);
        tensor_t (const tensor_t& another);
        tensor_t<T> operator + (tensor_t<T>& another);
        tensor_t<T> operator - (tensor_t<T>& another);

        T& operator () (int _x, int _y, int _z) {
            return data[_z * (this->size.x * this->size.y) + _y * this->size.x + _x];
        }

        T& get (int _x, int _y, int _z) {
            return data[_z * (this->size.x * this->size.y) + _y * this->size.x + _x];
        }

        ~tensor_t () {
            delete[] this->data;
        }
};
