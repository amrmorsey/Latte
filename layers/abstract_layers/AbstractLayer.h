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
public:
    const std::string &name;

    AbstractLayer(const std::string &name) : name(name) {};

    virtual ~AbstractLayer() {}

    Matrix calculateOutput(const Matrix &input_mat);
};

#endif //INFERENCEENGINE_LAYER_H
