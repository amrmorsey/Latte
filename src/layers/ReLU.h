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

    Matrix calculateOutput(const Matrix &input_mat) {};
};


#endif //INFERENCEENGINE_RELU_H
