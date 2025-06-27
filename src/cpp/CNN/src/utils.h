#pragma once
#include <cstdint>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "lib/modeling_cnn.h"
using namespace std;

struct case_t {
    tensor_t<float> data;
    tensor_t<float> out;
};

uint32_t byteswap_uint32 (uint32_t a);

uint8_t* read_file(const char* ffile);

std::vector<case_t> read_train_cases();

std::vector<case_t> read_test_cases();
