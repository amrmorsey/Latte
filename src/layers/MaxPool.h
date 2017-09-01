//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_MAXPOOLINGLAYER_H
#define INFERENCEENGINE_MAXPOOLINGLAYER_H

#include "abstract_layers/AbstractLayer.h"

class MaxPool : public AbstractLayer {
private:
    int kernel_size;
    int stride;
    int padding;
public:
    MaxPool(string name, int kernel_size, int stride, int padding) : AbstractLayer(name), kernel_size(kernel_size),
                                                                     stride(stride), padding(padding) {};

    ~MaxPool() {};

    Matrix calculateOutput(const Matrix &input_mat) {};
};


#endif //INFERENCEENGINE_MAXPOOLINGLAYER_H
