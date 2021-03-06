//
// Created by shadyf on 01/09/17.
//

#ifndef INFERENCEENGINE_RELU_H
#define INFERENCEENGINE_RELU_H


#include "abstract_layers/AbstractLayer.h"

class ReLU : public AbstractLayer {

public:
    explicit ReLU(std::string name) : AbstractLayer(name) {};

    ~ReLU() {};

// Calculates the output of the ReLU.
    void calculateOutput(MatrixAVX &input_mat) {
        __m256 zero = _mm256_setzero_ps();
        __m256 vcmp;
        __m256 x;
        float e;
        for (unsigned int i = 0; i < input_mat.xmm_size; ++i) {
            // Fetch a chunk from the matrix.
            x = input_mat.getChunk(i);
            // Check if any of its elements is greater than zero.
            vcmp = _mm256_cmp_ps(zero, x, _CMP_GT_OQ);
            auto vcmp_i = _mm256_cvtps_epi32(vcmp);
            if (!_mm256_testz_si256(vcmp_i, vcmp_i)) {
                // if there are elements set them all one at a time by checking if they are greater than zero, if yes put zero instead.
                output.setElement(i*8 + 0, std::max(input_mat.getElement(i * 8 + 0),0.0f));
                output.setElement(i*8 + 1, std::max(input_mat.getElement(i * 8 + 1),0.0f));
                output.setElement(i*8 + 2, std::max(input_mat.getElement(i * 8 + 2),0.0f));
                output.setElement(i*8 + 3, std::max(input_mat.getElement(i * 8 + 3),0.0f));
                output.setElement(i*8 + 4, std::max(input_mat.getElement(i * 8 + 4),0.0f));
                output.setElement(i*8 + 5, std::max(input_mat.getElement(i * 8 + 5),0.0f));
                output.setElement(i*8 + 6, std::max(input_mat.getElement(i * 8 + 6),0.0f));
                output.setElement(i*8 + 7, std::max(input_mat.getElement(i * 8 + 7),0.0f));
            } else
                output.setChunk(i, x);
        }
    };
// Sets up the ReLU layer, it takes the shape of the matrix before it to compute its own matrices.
    void precompute(std::vector<int> &in_mat) {
        output = MatrixAVX(in_mat);
    }

};


#endif //INFERENCEENGINE_RELU_H
