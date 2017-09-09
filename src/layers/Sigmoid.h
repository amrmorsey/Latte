//
// Created by shadyf on 02/09/17.
//

#ifndef INFERENCEENGINE_SIGMOID_H
#define INFERENCEENGINE_SIGMOID_H

#include "abstract_layers/AbstractLayer.h"

class Sigmoid : public AbstractLayer {
public:
    explicit Sigmoid(string name) : AbstractLayer(name) {};

    ~Sigmoid() {};

    void calculateOutput(Matrix &input_mat) {
        for (int i = 0; i < input_mat.matrix.size(); i++) {
            output.matrix[i] = input_mat.matrix[i] / (1 + abs(input_mat.matrix[i]));
        }
    };

    void precompute(Matrix& a){
        output = Matrix(a.shape);
    }
};

#endif //INFERENCEENGINE_SIGMOID_H
