//
// Created by shadyf on 01/09/17.
//

#ifndef INFERENCEENGINE_RELU_H
#define INFERENCEENGINE_RELU_H


#include "abstract_layers/AbstractLayer.h"

class ReLU : public AbstractLayer {

private:
    __m256 zero = _mm256_setzero_ps();
    __m256 vcmp;
public:
    explicit ReLU(std::string name) : AbstractLayer(name) {};

    ~ReLU() {};

    // use get and set chunks
    void calculateOutput(MatrixAVX &input_mat) {
//        for (unsigned int i = 0; i < input_mat.size; i++) {
//            float x = input_mat.getElement(i);
//            if ( x > 0)
//                output.setElement(i, x);
//        }
        __m256 x;
        float e;
        for (unsigned int i = 0; i < input_mat.xmm_size; ++i) {
             x = input_mat.getChunk(i);
            vcmp = _mm256_cmp_ps(zero, x, _CMP_GT_OQ);
            if (!_mm256_testz_si256(vcmp, vcmp)) {
                for (int j = 0; j < 8; ++j) {
                    e = input_mat.getElement(i*8 + j);
                    if ( e > 0)
                        output.setElement(i*8 + j, e);
                }
            }
            else
                output.setChunk(i, x);
        }
    };

    void precompute(MatrixAVX& in_mat){
        output = MatrixAVX(in_mat.shape);
    }

};


#endif //INFERENCEENGINE_RELU_H
