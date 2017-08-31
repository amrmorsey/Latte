//
// Created by shadyf on 29/08/17.
//

#ifndef INFERENCEENGINE_POOLING_LAYER_H
#define INFERENCEENGINE_POOLING_LAYER_H

#include <string>
#include <vector>
#include "AbstractLayer.h"

class AbstractPoolingLayer : AbstractLayer {
private:
    const int &kernel_size;
    const int &stride;
    const int &padding;
//    double *input;
//    double *output;
//    std::vector<int> input_dims;
//    std::vector<int> output_dims;
public:
    AbstractPoolingLayer(const string &name, const int &kernel_size, const int &stride, const int &padding)
            : AbstractLayer(name), kernel_size(kernel_size),
              stride(stride),
              padding(padding) {};

    ~AbstractPoolingLayer() {};
};


#endif //INFERENCEENGINE_POOLING_LAYER_H
