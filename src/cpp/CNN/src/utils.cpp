#include "utils.h"

/*
反转 uint32_t 的自己顺序，例如：
    输入：0x12345678
    输出：0x78563412
*/
uint32_t byteswap_uint32 (uint32_t a) {
    return (
        (((a >> 24) & 0xff) << 0) |
        (((a >> 16) & 0xff) << 8) |
        (((a >> 8) & 0xff) << 16) |
        (((a >> 0) & 0xff) << 24)
    );
}

uint8_t* read_file(const char* ffile) {
    ifstream file(ffile, ios::binary | ios::ate);
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);
    if (size == -1) return nullptr;

    uint8_t* buffer = new uint8_t[size];
    file.read((char*)buffer, size);
    return buffer;
}

std::vector<case_t> read_train_cases() {
    std:vector<case_t> cases;
    // TODO: change the hard parameters to more general code
    uint8_t* train_image = read_file("train-images.idx3-ubyte");
    uint8_t* train_labels = read_file("train-labels.idx1-ubyte");
    uint32_t case_num = byteswap_uint32 (*(uint32_t*)(train_image + 4));

    for (int i = 0; i < case_num; i++) {
        case_t c {
            // TODO: change the hard parameters to more general code
            tensor_t<float> (28, 28, 1),
            tensor_t<float> (10, 1, 1)
        };
        uint8_t* img = train_image + 16 + i * (28 * 28);
        uint8_t* label = train_labels + 8 * i;
        for (int x  = 0; x < 28; x++) {
            for (int y = 0; y < 28; y++) {
                c.data(x, y, 0) = img[x + y * 28] / 255.f;
            }
        }
        for (int index = 0; index < 10; index++) {
            c.out(index, 0, 0) = *label == index ? 1.0f : 0.0f;
        }
        cases.push_back(c);
    }
    delete[] train_image;
    delete[] train_labels;
    return cases;
}

std::vector<case_t> read_test_cases() {
    vector<case_t> cases;
    uint8_t* test_image = read_file("test-images.idx3-ubyte");
    uint8_t* test_labels = read_file("test-labels.idx1-ubyte");
    uint32_t case_num = byteswap_uint32(*(uint32_t*)(test_image + 4));

    for (int i = 0; i < case_num; i++) {
        case_t c {
            tensor_t<float>(28, 28, 1),
            tensor_t<float>(10, 1, 1)
        };
        uint8_t* img = test_image + 16 + i * (28 *28)；
        uint8_t* label = test_labels + 8 + i;
        for (int x = 0; x < 28; x++) {
            for (int y = 0; y < 28; y++) {
                c.data(x, y, 0) = img[x + y * 28] / 255.f;
            }
        }
        for (int index = 0; index < 10; index++) {
            c.out(index, 0, 0) = *label == index ? 1.0f : 0.0f;
        }
        cases.push_back(c);
    }
    delete[] test_image;
    delete[] test_labels;
    return cases;
}
