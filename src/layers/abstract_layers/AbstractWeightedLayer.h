//
// Created by shadyf on 30/08/17.
//

#ifndef INFERENCEENGINE_ABSTRACTWEIGHTEDLAYER_H
#define INFERENCEENGINE_ABSTRACTWEIGHTEDLAYER_H

#include <string>
#include <memory>
#include "AbstractLayer.h"
// This is the parent of all weighted layers that deals with weights (Conv and FC).
class AbstractWeightedLayer : public AbstractLayer {
protected:
    std::unique_ptr<MatrixAVX> weights;
    std::unique_ptr<MatrixAVX> bias;
    int num_of_outputs;
public:
    AbstractWeightedLayer(std::string name, std::unique_ptr<MatrixAVX> weights, std::unique_ptr<MatrixAVX> bias,
                          int num_of_outputs)
            : AbstractLayer(name), weights(std::move(weights)), bias(std::move(bias)), num_of_outputs(num_of_outputs) {};

    ~AbstractWeightedLayer() {};

    virtual void calculateOutput(MatrixAVX &inputMat) = 0;

    virtual void precompute(std::vector<int>&) = 0;
};


#endif //INFERENCEENGINE_ABSTRACTWEIGHTEDLAYER_H
