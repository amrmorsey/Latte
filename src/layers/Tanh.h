//
// Created by shadyf on 12/09/17.
//

#ifndef INFERENCEENGINE_TANH_H
#define INFERENCEENGINE_TANH_H

#include "abstract_layers/AbstractLayer.h"
#include <math.h>

class Tanh : public AbstractLayer {
public:
    explicit Tanh(std::string name) : AbstractLayer(name) {};

    ~Tanh() {};

    void calculateOutput(MatrixAVX &input_mat) {
        for (unsigned int i = 0; i < input_mat.size; ++i) {
            output.setElement(i, tanh(input_mat.getElement(i)));
        }
    };

    void precompute(std::vector<int>& in_mat){
        output = MatrixAVX(in_mat);
    }
};
#endif //INFERENCEENGINE_TANH_H
