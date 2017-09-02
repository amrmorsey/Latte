//
// Created by Amr on 8/21/17.
//

#ifndef INFERENCEENGINE_CONVLAYER_H
#define INFERENCEENGINE_CONVLAYER_H

#include <string>
#include <memory>
#include "abstract_layers/AbstractLayer.h"
#include "abstract_layers/AbstractWeightedLayer.h"

class ConvLayer : public AbstractWeightedLayer {

private:
    int kernel_size;
    int stride;
    int padding;

public:
    ConvLayer(std::string name, int num_of_outputs, std::unique_ptr<Matrix> weights, std::unique_ptr<Matrix> bias,
              int kernel_size, int stride, int padding) : AbstractWeightedLayer(name, std::move(weights),
                                                                                std::move(bias),
                                                                                num_of_outputs),
                                                          kernel_size(kernel_size), stride(stride),
                                                          padding(padding) {};

    ~ConvLayer() {};

    void calculateOutput(Matrix &input_mat) {
        input_mat = input_mat.conv(this->weights.get(), this->stride, this->padding);
    };
};


#endif //INFERENCEENGINE_CONVLAYER_H
