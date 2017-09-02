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

    void calculateOutput(Matrix &input_mat) {
        for (int i = 0; i < input_mat.matrix.size(); i++)
            if (input_mat.matrix[i] < 0)
                input_mat.matrix[i] = 0;
    };
};


#endif //INFERENCEENGINE_RELU_H
