//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_LAYER_H
#define INFERENCEENGINE_LAYER_H

#include <vector>
#include "../VecNN.h"
#include "../MatrixNN.h"
using namespace std;

class Layer {
public:
    void feedForward();
    void setInput(double*);
};


#endif //INFERENCEENGINE_LAYER_H
