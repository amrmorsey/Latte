//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_INPUTLAYER_H
#define INFERENCEENGINE_INPUTLAYER_H

#include "abstract_layers/AbstractLayer.h"

class InputLayer: public AbstractLayer {
private:
    vector<int> input_dim;
public:
    InputLayer(std::string name, vector<int> input_dim) : AbstractLayer(name), input_dim(input_dim) {};
    ~InputLayer() {};
    Matrix calculateOutput(const Matrix &input_mat) {};
};


#endif //INFERENCEENGINE_INPUTLAYER_H
