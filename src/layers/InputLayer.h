//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_INPUTLAYER_H
#define INFERENCEENGINE_INPUTLAYER_H

#include <exception>
#include "abstract_layers/AbstractLayer.h"

class InputLayer : public AbstractLayer {
private:
    std::vector<int> input_dim;
public:
    InputLayer(std::string name, std::vector<int> input_dim) : AbstractLayer(name), input_dim(input_dim) {};

    ~InputLayer() {};

    void calculateOutput(MatrixAVX &input_mat) {
        int input_size = 1;

        for (int dim : input_dim)
            input_size *= dim;

        if (input_size != input_mat.size)
            throw std::length_error("Input image dimensions does not match dimensions of input layer");
    };
};


#endif //INFERENCEENGINE_INPUTLAYER_H
