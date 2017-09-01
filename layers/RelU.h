//
// Created by shadyf on 01/09/17.
//

#ifndef INFERENCEENGINE_RELU_H
#define INFERENCEENGINE_RELU_H


#include "abstract_layers/AbstractLayer.h"

class RelU : public AbstractLayer {
public:
    explicit RelU(std::string name) : AbstractLayer(name) {};

    ~RelU() {};

    Matrix calculateOutput(const Matrix &input_mat) {};
};


#endif //INFERENCEENGINE_RELU_H
