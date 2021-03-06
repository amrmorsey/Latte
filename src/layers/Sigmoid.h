//
// Created by shadyf on 02/09/17.
//

#ifndef INFERENCEENGINE_SIGMOID_H
#define INFERENCEENGINE_SIGMOID_H

#include "abstract_layers/AbstractLayer.h"

class Sigmoid : public AbstractLayer {
public:
    explicit Sigmoid(std::string name) : AbstractLayer(name) {};

    ~Sigmoid() {};
// Calculates the output of the Sigmoid.
    void calculateOutput(MatrixAVX &input_mat) {
        for (unsigned int i = 0; i < input_mat.size; ++i) {
            output.setElement(i, std::exp(input_mat.getElement(i)));
        }
        float x;
        for (unsigned int i = 0; i < input_mat.size; ++i) {
            x = output.getElement(i);
            output.setElement(i,  x/ (x+1));
        }
    };
// Sets up the Sigmoid layer, it takes the shape of the matrix before it to compute its own matrices.
    void precompute(std::vector<int>& in_mat){
        output = MatrixAVX(in_mat);
    }
};

#endif //INFERENCEENGINE_SIGMOID_H
