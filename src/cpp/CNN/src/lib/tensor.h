#pragma once
#include <cassert>
#include <vector>
#include <cstring>

typedef struct dims_t {
    int x, y, z;
} td_size;

template<typename T>
class tensor_t {
    public:
        T* data;
        td_size size;
        tensor_t (int _x, int _y, int _z) {
            this->data = new T[_x * _y * _z];
            this->size.x = _x;
            this->size.y = _y;
            this->size.z = _z;
        }

        tensor_t (const tensor_t& another) {
            this->data = new T[another.size.x * another.size.y * another.size.z];
            memcpy(this->data, another.data, another.size.x * another.size.y * another.size.z * sizeof(T));
            this->size = another.size;
        }
        tensor_t<T> operator + (tensor_t<T>& another) {
            tensor_t<T> ret(*this);
            for (int i = 0; i < another.size.x * another.size.y * another.size.z; i++) {
                ret.data[i] += another.data[i];
            }
            return ret;
        }
        tensor_t<T> operator - (tensor_t<T>& another) {
            tensor_t<T> ret(*this);
            for (int i = 0; i < another.size.x * another.size.y * another.size.z; i++) {
                ret.data[i] -= another.data[i];
            }
            return ret;
        }


        T& operator () (int _x, int _y, int _z) {
            return data[_z * (this->size.x * this->size.y) + _y * this->size.x + _x];
        }

        T& get (int _x, int _y, int _z) {
            return data[_z * (this->size.x * this->size.y) + _y * this->size.x + _x];
        }

        ~tensor_t () {
            if (this->data) delete[] this->data;
        }
};
