//
// Created by Amr on 8/21/17.
//

#ifndef INFERENCEENGINE_CONVLAYER_H
#define INFERENCEENGINE_CONVLAYER_H

#include <string>
#include "abstract_layers/AbstractLayer.h"
#include "abstract_layers/AbstractWeightedLayer.h"

using namespace std;

class ConvLayer : public AbstractWeightedLayer {

private:
    const int &kernel_size;
    const int &stride;
    const int &padding;

public:
    ConvLayer(const std::string &name, const int &num_of_outputs, const Matrix &weights, const Matrix &bias,
              const int &kernel_size, const int &stride, const int &padding) : AbstractWeightedLayer(name, weights,
                                                                                                     bias,
                                                                                                     num_of_outputs),
                                                                               kernel_size(kernel_size), stride(stride),
                                                                               padding(padding) {};

    ~ConvLayer() {};

};


#endif //INFERENCEENGINE_CONVLAYER_H
