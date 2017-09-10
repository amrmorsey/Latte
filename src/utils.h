//
// Created by shadyf on 08/09/17.
//

#ifndef INFERENCEENGINE_UTILS_H
#define INFERENCEENGINE_UTILS_H

#include <string>
#include <vector>
#include <fstream>
#include "MatrixAVX.h"

std::vector<float> extractValues(const std::string &file_path);

MatrixAVX loadMatrix(const std::string &matrix_dir, const std::string &matrix_name);

void im2col(MatrixAVX &input_mat, const std::vector<int> &filterShape, MatrixAVX &out, int s, int pad, int x);

inline float dot_product(const __m128 &a, const __m128 &b) {
    __m128 r1 = _mm_mul_ps(a, b);
    __m128 r2 = _mm_hadd_ps(r1, r1);
    __m128 r3 = _mm_hadd_ps(r2, r2);
    return float(r3[0]);
//    return float(MatrixAVX::hsums(_mm256_mul_ps(a, b))[0]);
}

#endif //INFERENCEENGINE_UTILS_H
