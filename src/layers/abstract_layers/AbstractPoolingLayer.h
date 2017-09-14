//
// Created by shadyf on 29/08/17.
//

#ifndef INFERENCEENGINE_POOLING_LAYER_H
#define INFERENCEENGINE_POOLING_LAYER_H

#include <string>
#include <vector>
#include "AbstractLayer.h"
// This is the parent of the pooling layer, it has no weights, only a kernel, padding and stride.
class AbstractPoolingLayer : AbstractLayer {
private:
    int kernel_size;
    int stride;
    int padding;
public:
    AbstractPoolingLayer(std::string name, int kernel_size, int stride, int padding)
            : AbstractLayer(name), kernel_size(kernel_size),
              stride(stride),
              padding(padding) {};

    ~AbstractPoolingLayer() {};

    virtual void precompute(std::vector<int>&) = 0;
};


#endif //INFERENCEENGINE_POOLING_LAYER_H
