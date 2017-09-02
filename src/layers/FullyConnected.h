//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_FULLYCONNECTEDLAYER_H
#define INFERENCEENGINE_FULLYCONNECTEDLAYER_H

#include "abstract_layers/AbstractLayer.h"
#include "abstract_layers/AbstractWeightedLayer.h"

class FullyConnected : public AbstractWeightedLayer {
public:
    FullyConnected(string name, int num_of_outputs, std::unique_ptr<Matrix> weights, std::unique_ptr<Matrix> bias)
            : AbstractWeightedLayer(name, std::move(weights), std::move(bias), num_of_outputs) {};

    ~FullyConnected() {};

    void calculateOutput(Matrix &input_mat) {
        this->weights->shape = {1, 1, input_mat.shape[0], 1};
        input_mat.conv(this->weights.get(), 1, 0);
       // this->weights->conv(&input_mat,1,0);
    };
};


#endif //INFERENCEENGINE_FULLYCONNECTEDLAYER_H
