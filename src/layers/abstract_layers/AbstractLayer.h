//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_LAYER_H
#define INFERENCEENGINE_LAYER_H

#include <vector>
#include <string>
#include "../../MatrixAVX.h"

class AbstractLayer {
public:
    std::string name;
    MatrixAVX input;
    MatrixAVX output;
    MatrixAVX output_before_bias;
    MatrixAVX im2col_out;
    aligned_vector biases;
    aligned_vector biases_stranglers;
    int rem;

    explicit AbstractLayer(std::string name) : name(name) {};

    virtual ~AbstractLayer() = default;

    virtual void calculateOutput(MatrixAVX &input_mat) = 0;

    virtual void precompute(MatrixAVX&) = 0;
};

#endif //INFERENCEENGINE_LAYER_H
