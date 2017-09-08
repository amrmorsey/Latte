#ifndef INFERENCEENGINE_MATRIXAVX_H
#define INFERENCEENGINE_MATRIXAVX_H

#ifdef _WIN32
#include <malloc.h>
#endif

#include "AlignedVector.h"

#include <avxintrin.h>
#include <xmmintrin.h>
#include <vector>
#include <cmath>
#include <exception>
#include <string>
#include <algorithm>
#include <ostream>

class MatrixAVX {
private:
    aligned_vector xmm;
    unsigned long xmm_size;
    unsigned long aligned_size;
    unsigned int stranglers;

    // TODO: Change this to support windows
    __attribute__((aligned(sizeof(__m256)))) float aligned_float_arr[8];

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

public:
    unsigned long size;
    std::vector<int> shape;

    explicit MatrixAVX(std::vector<float> vec, std::vector<int> shape) : shape(shape) {
        size = 1;

        for (int x : shape)
            size *= x;

        xmm_size = static_cast<unsigned long>(ceil(size / 8.0f));

        aligned_size = size / 8;

        for (int i = 0; i < aligned_size; i++) {
            xmm.push_back(_mm256_loadu_ps(&vec[i * 8]));
        }

        // Check for stranglers in case matrix size is not divisible by 8
        stranglers = static_cast<unsigned int>(size % 8);
        if (stranglers) {
            // Set up a mask for partial loading.
            // Highest bit -> 1 in mask element means corresponding array element will be taken (i.e negative values)
            // Highest bit -> 0 in mask element means corresponding array element will be taken as 0
            unsigned int rem = stranglers;
            __m256i mask = _mm256_setr_epi32(-rem, 1 - rem, 2 - rem, 3 - rem, 4 - rem, 5 - rem, 6 - rem, 7 - rem);
            xmm.push_back(_mm256_maskload_ps(&vec[aligned_size * 8], mask));
        }
    };

    explicit MatrixAVX(std::vector<int> shape) : shape(shape), aligned_size(0), stranglers(0) {
        size = 1;

        for (int x : shape)
            size *= x;

        xmm_size = static_cast<unsigned long>(ceil(size / 8.0f));


        for (int i = 0; i < xmm_size; i++) {
            xmm.push_back(_mm256_setzero_ps());
        }
    }

    __m256 operator[](unsigned int index) const {
        return getChunk(index);
    }

    // Get a single element (float value) from the matrix
    inline float getElement(unsigned int index) {
        _mm256_store_ps(aligned_float_arr, xmm[index / 8]);
        return aligned_float_arr[index % 8];
    }

    // Set a single element (float value) into the matrix
    // Caution: This might be an expensive operation if called multiple times. Use setChunk instead
    inline void setElement(unsigned int index, float value) {
        if (index >= size) {
            throw std::out_of_range("Index " + std::to_string(index) + " is out of range. Matrix size is " +
                                    std::to_string(size));
        }
        _mm256_store_ps(aligned_float_arr, xmm[index / 8]);
        aligned_float_arr[index % 8] = value;
        xmm[index / 8] = _mm256_load_ps(aligned_float_arr);
    }

    // Set a whole chunk (8 float values) into the matrix
    // This is prefered over setElement
    inline void setChunk(unsigned int index, __m256 chunk) {
        if (index >= xmm_size) {
            throw std::out_of_range(
                    "Index " + std::to_string(index) + " is out of range. Total number of chunks is " +
                    std::to_string(xmm_size));
        }
        xmm[index] = chunk;
    }

    // Retrieve a chunk from the matrix
    __m256 getChunk(unsigned int index) const {
        return xmm[index];
    }

    void add(const MatrixAVX &a, MatrixAVX &out) {
        if (size != a.size) {
            throw std::logic_error(
                    "Matrices not of equal size (" + std::to_string(size) + ") vs (" + std::to_string(a.size) +
                    ")");
        }
        for (unsigned int i = 0; i < xmm_size; i++) {
            out.setChunk(i, _mm256_add_ps(xmm[i], a.xmm[i]));
        }
    }

    void sub(const MatrixAVX &a, MatrixAVX &out) {
        if (size != a.size) {
            throw std::logic_error(
                    "Matrices not of equal size (" + std::to_string(size) + ") vs (" + std::to_string(a.size) +
                    ")");
        }
        for (unsigned int i = 0; i < xmm_size; i++) {
            out.setChunk(i, _mm256_sub_ps(xmm[i], a.xmm[i]));
        }
    }

    // Sub that takes a single value instead of an entire matrix
    void sub(const float &a, MatrixAVX &out) {
        __m256 sub_chunk = _mm256_set1_ps(a);
        for (unsigned int i = 0; i < xmm_size; i++) {
            out.setChunk(i, _mm256_sub_ps(xmm[i], sub_chunk));
        }
    }

    // Calculates dot product of two matricies
    // Out is expected to be initialized with its xmm vector already resize to the correct length
    void dot_product(const MatrixAVX &a, MatrixAVX &out) {
        if (size != a.size) {
            throw std::logic_error(
                    "Matrices not of equal size (" + std::to_string(size) + ") vs (" + std::to_string(a.size) +
                    ")");
        }

        std::fill(aligned_float_arr, aligned_float_arr + 8, 0);

        unsigned int out_index = 0;

        for (unsigned int i = 1; i <= xmm_size; i++) {
            if (i % 8 == 0) {
                out.setChunk(out_index++, _mm256_load_ps(aligned_float_arr));
                std::fill(aligned_float_arr, aligned_float_arr + 8, 0);
            }
            aligned_float_arr[i - 1 % 8] = float(hsums(_mm256_mul_ps(xmm[i - 1], a.xmm[i - 1]))[0]);
        }
        out.setChunk(out_index, _mm256_load_ps(aligned_float_arr));
    }

    void reshape(const std::vector<int> &new_shape) {
        unsigned long new_size = 1;
        std::string shape_str = "(";

        for (int x : new_shape) {
            shape_str += x + " ";
            new_size *= x;
        }
        shape_str += ")";

        if (size != new_size)
            throw std::logic_error(
                    "Cannot reshape matrix of size " + std::to_string(size) + " into shape " + shape_str);

        shape = new_shape;
    }

    // operator << : Displays contents of matrix
    friend std::ostream &operator<<(std::ostream &stream, const MatrixAVX &matrix) {
        for (unsigned int i = 0; i < matrix.xmm_size; i++) {
            stream << "[";
            for (unsigned int j = 0; j < 7; j++)
                stream << std::to_string(matrix.xmm[i][j]) + " ";
            stream << std::to_string(matrix.xmm[i][7]);
            stream << "]\n";
        }

    }

};

#endif //INFERENCEENGINE_MATRIXAVX_H
