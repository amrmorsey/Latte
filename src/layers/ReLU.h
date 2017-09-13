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

    // use get and set chunks
    void calculateOutput(MatrixAVX &input_mat) {
        __m256 zero;
        __m256 vcmp;
        __m256 x;
        float e;
        for (unsigned int i = 0; i < input_mat.xmm_size; ++i) {
            x = input_mat.getChunk(i);
            vcmp = _mm256_cmp_ps(zero, x, _CMP_GT_OQ);
            auto vcmp_i = _mm256_cvtps_epi32(vcmp);
            if (!_mm256_testz_si256(vcmp_i, vcmp_i)) {
//                for (int j = 0; j < 8; ++j) {
//                    e = input_mat.getElement(i * 8 + j);
//                    if (e > 0)
//                        output.setElement(i * 8 + j, e);
//                }
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

    void precompute(MatrixAVX &in_mat) {
        output = MatrixAVX(in_mat.shape);
    }

};


#endif //INFERENCEENGINE_RELU_H
