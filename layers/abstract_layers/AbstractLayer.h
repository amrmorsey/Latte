//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_LAYER_H
#define INFERENCEENGINE_LAYER_H

#include <vector>
#include <string>
#include "../../VecNN.h"
#include "../../MatrixNN.h"
#include "../../Matrix.h"

class AbstractLayer {
private:
    Matrix weights;
    Matrix bias;

public:
    std::string name;

    AbstractLayer(const string &name, const Matrix &weights, const Matrix &bias)
            : name(name), weights(weights), bias(bias) {};

    ~AbstractLayer() {}

    void setWeights(const Matrix &weights);

    void setBias(const Matrix &bias);

    Matrix calculateOutput(const Matrix &inputMat);
};

#endif //INFERENCEENGINE_LAYER_H
