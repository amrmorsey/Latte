//
// Created by shadyf on 01/09/17.
//

#ifndef INFERENCEENGINE_SOFTMAX_H
#define INFERENCEENGINE_SOFTMAX_H


#include "abstract_layers/AbstractLayer.h"

class Softmax : public AbstractLayer {
public:
    explicit Softmax(string name) : AbstractLayer(name) {};

    ~Softmax() {};

    Matrix calculateOutput(const Matrix &input_mat) {};
};


#endif //INFERENCEENGINE_SOFTMAX_H
