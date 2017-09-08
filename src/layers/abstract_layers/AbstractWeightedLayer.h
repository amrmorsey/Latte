//
// Created by shadyf on 30/08/17.
//

#ifndef INFERENCEENGINE_ABSTRACTWEIGHTEDLAYER_H
#define INFERENCEENGINE_ABSTRACTWEIGHTEDLAYER_H

#include <string>
#include <memory>
#include "AbstractLayer.h"

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

    void setWeights(const MatrixAVX &weights);

    void setBias(const MatrixAVX &bias);

    virtual void calculateOutput(MatrixAVX &inputMat) = 0;
};


#endif //INFERENCEENGINE_ABSTRACTWEIGHTEDLAYER_H
