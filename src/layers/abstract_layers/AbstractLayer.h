//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_LAYER_H
#define INFERENCEENGINE_LAYER_H

#include <utility>
#include <vector>
#include <string>
#include "../../MatrixAVX.h"
//This is the abstract class for layer, its the parent of all classes.
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

    explicit AbstractLayer(std::string name) : name(std::move(name)) {};

    virtual ~AbstractLayer() = default;
//  Calculate output is the function that computes the output of this layer.
    virtual void calculateOutput(MatrixAVX &input_mat) = 0;
//  Precomute sets up the required matrices and variables required for calculateOutput to work.
    virtual void precompute(std::vector<int>&) = 0;
};

#endif //INFERENCEENGINE_LAYER_H
