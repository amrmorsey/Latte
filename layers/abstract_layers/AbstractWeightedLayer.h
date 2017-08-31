//
// Created by shadyf on 30/08/17.
//

#ifndef INFERENCEENGINE_ABSTRACTWEIGHTEDLAYER_H
#define INFERENCEENGINE_ABSTRACTWEIGHTEDLAYER_H

#include <string>
#include "../../Matrix.h"
#include "AbstractLayer.h"

class AbstractWeightedLayer : AbstractLayer {
private:
    const Matrix &weights;
    const Matrix &bias;
    const int &num_of_outputs;
public:
    AbstractWeightedLayer(const std::string &name, const Matrix &weights, const Matrix &bias, const int &num_of_outputs)
            : AbstractLayer(name), weights(weights), bias(bias), num_of_outputs(num_of_outputs) {};

    ~AbstractWeightedLayer() {};

    void setWeights(const Matrix &weights);

    void setBias(const Matrix &bias);

    Matrix calculateOutput(const Matrix &inputMat);
};


#endif //INFERENCEENGINE_ABSTRACTWEIGHTEDLAYER_H
