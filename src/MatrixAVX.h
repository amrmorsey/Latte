//
// Created by shadyf on 07/09/17.
//

#ifndef INFERENCEENGINE_MATRIXAVX_H
#define INFERENCEENGINE_MATRIXAVX_H


#include <avxintrin.h>
#include <xmmintrin.h>
#include <vector>
#include <cmath>

class MatrixAVX {
private:

//    https://stackoverflow.com/questions/13879609/horizontal-sum-of-8-packed-32bit-floats
    static inline __m256 hsums(__m256 const &v) {
        auto x = _mm256_permute2f128_ps(v, v, 1);
        auto y = _mm256_add_ps(v, x);
        x = _mm256_shuffle_ps(y, y, _MM_SHUFFLE(2, 3, 0, 1));
        x = _mm256_add_ps(x, y);
        y = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
        return _mm256_add_ps(x, y);
    }

    __m256 *xmm;
    unsigned long matrix_size;
    std::vector<int> matrix_shape;
    unsigned long xmm_size;
public:

    explicit MatrixAVX(std::vector<float> vec, std::vector<int> shape) {
        matrix_size = 0;

        for (int x : shape)
            matrix_size += x;

        xmm_size = static_cast<unsigned long>(ceil(matrix_size / 8.0f));
//        xmm = _mm_malloc(xmm_size, 16);
        for (int i = 0; i < xmm_size; i++) {
            xmm[i] = _mm256_loadu_ps(&vec[i * 8]);
        }
    };
    ~MatrixAVX() {
        _mm_free(xmm);
    }
    static void dot_product(MatrixAVX const &a, MatrixAVX &out) {
        __m256 evens = _mm256_set_ps(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        __m256 odds = _mm256_set_ps(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 6.0);
        __m256 c = _mm256_mul_ps(evens, odds);
    }

};


#endif //INFERENCEENGINE_MATRIXAVX_H
