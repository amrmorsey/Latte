//
// Created by shadyf on 01/09/17.
//

#ifndef INFERENCEENGINE_SOFTMAX_H
#define INFERENCEENGINE_SOFTMAX_H


#include "abstract_layers/AbstractLayer.h"

class Softmax : public AbstractLayer {
public:
    explicit Softmax(std::string name) : AbstractLayer(name) {};

    ~Softmax() {};

    // Can this calculation be done inplace?
    void calculateOutput(MatrixAVX &input_mat) {
        softMaxFunction3(input_mat);
    };

    void softMaxFunction3(MatrixAVX &input_mat){
        float sum = 0;
        for (unsigned int i = 0; i < input_mat.size; ++i) {
            output.setElement(i, std::exp(input_mat.getElement(i)));
            sum += output.getElement(i);
        }

        for (unsigned int i = 0; i < input_mat.size; ++i) {
            output.setElement(i, output.getElement(i) / sum);
        }
    }

    void precompute(MatrixAVX& in_mat){
        output = MatrixAVX(in_mat.shape);
    }

};


#endif //INFERENCEENGINE_SOFTMAX_H
