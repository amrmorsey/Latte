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
    MatrixAVX filter;
    std::vector<float> biases;
    MatrixAVX biasMat;
    MatrixAVX smaller_mat;
    unsigned int chunk_range;
    unsigned int big_reserve_size;
    unsigned int small_reserve_size;
    MatrixAVX s,b;
    std::vector<float> big_matrix_vec;
    std::vector<float> small_matrix_vec;
    int repeated_dim, kept_dim, other_dim;


    int rem;

    explicit AbstractLayer(std::string name) : name(name) {};

    virtual ~AbstractLayer() = default;

    virtual void calculateOutput(MatrixAVX &input_mat) = 0;

    virtual void precompute(MatrixAVX&) = 0;
};

#endif //INFERENCEENGINE_LAYER_H
