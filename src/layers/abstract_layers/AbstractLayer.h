//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_LAYER_H
#define INFERENCEENGINE_LAYER_H

#include <vector>
#include <string>
#include "../../VecNN.h"
#include "../../Matrix.h"

class AbstractLayer {
public:
    std::string name;

    explicit AbstractLayer(std::string name) : name(name) {};

    virtual ~AbstractLayer() = default;

    virtual Matrix calculateOutput(const Matrix &input_mat) = 0;
};

#endif //INFERENCEENGINE_LAYER_H
