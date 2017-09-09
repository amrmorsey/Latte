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
            if (input_mat.matrix[i] > 0)
                output.matrix[i] = input_mat.matrix[i];
        //output.matrix = input_mat.matrix;
    };

    void precompute(Matrix& a){
        output = Matrix(a.shape);
    }

};


#endif //INFERENCEENGINE_RELU_H
