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

// Calculates the output of the InputLayer.
    void calculateOutput(MatrixAVX &input_mat) {
        int input_size = 1;
        //Check input image dim to see if it is the required dimensions.
        for (int dim : input_dim)
            input_size *= dim;
        //otherwise print an error.
        if (input_size != input_mat.size)
            throw std::length_error("Input image dimensions does not match dimensions of input layer");
        //set the output to be equal to the input.
        output = input_mat;
    };
// Sets up the InputLayer layer, it takes the shape of the matrix before it to compute its own matrices.
    void precompute(std::vector<int>& in_mat){
        input = MatrixAVX(in_mat);
        output = input;
    }
};


#endif //INFERENCEENGINE_INPUTLAYER_H
