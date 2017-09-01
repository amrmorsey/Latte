//
// Created by shadyf on 30/08/17.
//

#ifndef INFERENCEENGINE_ABSTRACTWEIGHTEDLAYER_H
#define INFERENCEENGINE_ABSTRACTWEIGHTEDLAYER_H

#include <string>
#include <memory>
#include "../../Matrix.h"
#include "AbstractLayer.h"

class AbstractWeightedLayer : public AbstractLayer {
private:
    std::unique_ptr<Matrix> weights;
    std::unique_ptr<Matrix> bias;
    int num_of_outputs;
public:
    AbstractWeightedLayer(std::string name, std::unique_ptr<Matrix> weights, std::unique_ptr<Matrix> bias,
                          int num_of_outputs)
            : AbstractLayer(name), weights(std::move(weights)), bias(std::move(bias)), num_of_outputs(num_of_outputs) {};

    ~AbstractWeightedLayer() {};

    void setWeights(const Matrix &weights);

    void setBias(const Matrix &bias);

    virtual Matrix calculateOutput(const Matrix &inputMat) = 0;
};


#endif //INFERENCEENGINE_ABSTRACTWEIGHTEDLAYER_H
