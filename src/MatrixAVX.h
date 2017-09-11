#ifndef INFERENCEENGINE_MATRIXAVX_H
#define INFERENCEENGINE_MATRIXAVX_H

#ifdef _WIN32
#include <malloc.h>
#endif

#include "AlignedVector.h"

#include <immintrin.h>
#include <utility>
#include <vector>
#include <cmath>
#include <exception>
#include <string>
#include <algorithm>
#include <ostream>
#include <iostream>
#include <iomanip>

union __m256_f{
    __m256 v;
    float f[8];
};

typedef std::vector<__m256_f, aligned_allocator<__m256_f, sizeof(__m256_f)> > aligned_vector;

class MatrixAVX {
private:

    unsigned long xmm_size;
    unsigned long aligned_size;
    unsigned int stranglers;

public:
    aligned_vector xmm;
    std::vector<int> X_col_shape;
    std::vector<int> W_row_shape;
    unsigned long size;
    std::vector<int> shape;

    explicit MatrixAVX(const std::vector<float> &vec, std::vector<int> shape) : shape(shape) {
        size = 1;

        for (int x : shape)
            size *= x;

        xmm_size = static_cast<unsigned long>(std::ceil(size / 8.0f));

        aligned_size = size / 8;
        xmm.resize(aligned_size + 1);

        for (int i = 0; i < aligned_size; i++) {
            xmm[i].v = _mm256_loadu_ps(&vec[i*8]);
        }

        // Check for stranglers in case matrix size is not divisible by 8
        stranglers = static_cast<unsigned int>(size % 8);
        if (stranglers) {
            // Set up a mask for partial loading.
            // Highest bit -> 1 in mask element means corresponding array element will be taken (i.e negative values)
            // Highest bit -> 0 in mask element means corresponding array element will be taken as 0
            unsigned int rem = stranglers;
            __m256i mask = _mm256_setr_epi32(-rem, 1 - rem, 2 - rem, 3 - rem, 4 - rem, 5 - rem, 6 - rem, 7 - rem);
            xmm[aligned_size].v = _mm256_maskload_ps(&vec[aligned_size * 8], mask);
        }
    };

    explicit MatrixAVX(std::vector<int> shape) : shape(shape), aligned_size(0), stranglers(0) {
        size = 1;

        for (int x : shape)
            size *= x;

        xmm_size = static_cast<unsigned long>(ceil(size / 8.0f));

        xmm.resize(xmm_size);

        for (int i = 0; i < xmm_size; i++) {
            xmm[i].v = _mm256_setzero_ps();
        }
    }

    explicit MatrixAVX(aligned_vector xmm, std::vector<int> shape) : shape(shape), xmm(xmm), aligned_size(0),
                                                                     stranglers(0) {
        size = 1;

        for (int x : shape)
            size *= x;

        xmm_size = static_cast<unsigned long>(ceil(size / 8.0f));
    }

    MatrixAVX() : shape({0}), aligned_size(0), stranglers(0), size(0) {}

    __m256 operator[](unsigned int index) const {
        return getChunk(index);
    }

    // Get a single element (float value) from the matrix
    inline float getElement(unsigned int index) {
        return xmm[index / 8].f[index % 8];
    }

    // Set a single element (float value) into the matrix
    // Caution: This might be an expensive operation if called multiple times. Use setChunk instead
    inline void setElement(unsigned int index, float value) {
//        if (index >= xmm_size * 8) {
//            throw std::out_of_range("Index " + std::to_string(index) + " is out of range. Matrix size is " +
//                                    std::to_string(size));
//        }
        xmm[index / 8].f[index % 8] = value;
    }

    // Set a whole chunk (8 float values) into the matrix
    // This is prefered over setElement
    inline void setChunk(unsigned int index, __m256 chunk) {
//        if (index >= xmm_size) {
//            throw std::out_of_range(
//                    "Index " + std::to_string(index) + " is out of range. Total number of chunks is " +
//                    std::to_string(xmm_size));
//        }
        xmm[index].v = chunk;
    }

    // Retrieve a chunk from the matrix
    __m256 getChunk(unsigned int index) const {
        return xmm[index].v;
    }

    void add(MatrixAVX bias, MatrixAVX &out) {
//        int limit = std::floor((this->shape[0] * this->shape[1]) / 8.0);
//        int rem = (this->shape[0] * this->shape[1]) % 8;
//        int j = 0;
//        for (unsigned int i = 0; i < biases.size(); i++) {
//            for (; j < limit + i * (limit + 1) && j < this->xmm_size; ++j) {
//                out.setChunk(j, _mm256_add_ps(xmm[j], biases[i]));
//            }
//            if (j < this->xmm_size && rem) {
//                out.setChunk(j, _mm256_add_ps(xmm[j], stranglers[i]));
//                ++j;
//            }
//        }
//        for (int i = 0; i < 8 - (out.size % 8); i++) {
//            out.setElement(out.xmm_size * 8 - 1 - i, 0.0f);
//        }
        for (int i = 0; i < bias.xmm_size; ++i) {
            out.setChunk(i, _mm256_add_ps(xmm[i].v, bias.xmm[i].v));
        }
    }

    void sub(const MatrixAVX &a, MatrixAVX &out) {
        if (size != a.size) {
            throw std::logic_error(
                    "Matrices not of equal size (" + std::to_string(size) + ") vs (" + std::to_string(a.size) +
                    ")");
        }
        for (unsigned int i = 0; i < xmm_size; i++) {
            out.setChunk(i, _mm256_sub_ps(xmm[i].v, a.xmm[i].v));
        }
    }

    // Sub that takes a single value instead of an entire matrix
    void sub(const float &a, MatrixAVX &out) {
        __m256 sub_chunk = _mm256_set1_ps(a);
        for (unsigned int i = 0; i < xmm_size; i++) {
            out.setChunk(i, _mm256_sub_ps(xmm[i].v, sub_chunk));
        }
    }

    // Calculates dot product of two matricies
    // Out is expected to be initialized with its xmm vector already resize to the correct length
    void dot_product(int kept_dim, const std::vector<float> &big_matrix_vec, unsigned int big_reserve_size, const MatrixAVX& small,unsigned int chunk_range, MatrixAVX &out) {
        int out_index = 0;
        float res;

        for (int small_chunk = 0; small_chunk < small.xmm_size; small_chunk += chunk_range) {
            for (int big_chunk = 0; big_chunk < xmm_size; big_chunk += chunk_range) {
                res = 0;
                for (int partial_index = 0; partial_index < chunk_range; partial_index++) {
                    // AVX2 float conversion is ~10-20microseconds faster
                    res += _mm_cvtss_f32(_mm256_castps256_ps128(hsums(_mm256_mul_ps(xmm[big_chunk + partial_index].v,
                                                     small.xmm[small_chunk + partial_index].v))));
                }
                out.setElement(static_cast<unsigned int>(out_index++), res);
            }
        }
    }

    std::string shape_str() const {
        std::string shape_str = "(";

        for (int i = 0; i < shape.size() - 1; i++) {
            shape_str += std::to_string(shape[i]) + ", ";
        }
        shape_str += std::to_string(shape[shape.size() - 1]) + ")";
        return shape_str;

    }

    void reshape(const std::vector<int> &new_shape) {
        unsigned long new_size = 1;

        for (int x : new_shape) {
            new_size *= x;
        }

//        if (size != new_size) {
//            std::string shape_str = "(";
//
//            for (int i = 0; i < new_shape.size() - 1; i++) {
//                shape_str += std::to_string(new_shape[i]) + ", ";
//            }
//            shape_str += std::to_string(new_shape[new_shape.size() - 1]) + ")";
//
//            throw std::logic_error(
//                    "Cannot reshape matrix of size " + std::to_string(size) + " into shape " + shape_str);
//
//        }
        shape = new_shape;
    }

    // Does horizontal sum of a chunk v
    // Only works if v is __m256, __m128 requires less instructions
    static inline __m256 hsums(__m256 const &v) {
        auto x = _mm256_permute2f128_ps(v, v, 1);
        auto y = _mm256_add_ps(v, x);
        x = _mm256_shuffle_ps(y, y, _MM_SHUFFLE(2, 3, 0, 1));
        x = _mm256_add_ps(x, y);
        y = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
        return _mm256_add_ps(x, y);
    };

    // operator << : Displays contents of matrix
    friend std::ostream &operator<<(std::ostream &stream, const MatrixAVX &matrix) {
        for (unsigned int i = 0; i < matrix.xmm_size; i++) {
            stream << std::to_string(i * 8) + " - [";
            for (unsigned int j = 0; j < 7; j++)
                stream << std::to_string(matrix.xmm[i].f[j]) + " ";
            stream << std::to_string(matrix.xmm[i].f[7]);
            stream << "]\n";
        }
        return stream;
    }
};

#endif //INFERENCEENGINE_MATRIXAVX_H
