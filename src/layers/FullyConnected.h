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

    Matrix calculateOutput(const Matrix &input_mat) {};
};


#endif //INFERENCEENGINE_FULLYCONNECTEDLAYER_H
