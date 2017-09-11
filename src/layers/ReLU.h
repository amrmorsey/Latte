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
        for (unsigned int i = 0; i < input_mat.size; i++) {
            float x = input_mat.getElement(i);
            if ( x > 0)
                output.setElement(i, x);
        }
    };

    void precompute(MatrixAVX& in_mat){
        output = MatrixAVX(in_mat.shape);
    }

};


#endif //INFERENCEENGINE_RELU_H
