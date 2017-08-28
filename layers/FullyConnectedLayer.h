//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_FULLYCONNECTEDLAYER_H
#define INFERENCEENGINE_FULLYCONNECTEDLAYER_H

#include "Layer.h"

class FullyConnectedLayer: public Layer {
private:
    int num_of_neurons;
    double *input;
    double *weights;
    double *output;
    vector<int> input_dims;
    vector<int> output_dims;
public:
    FullyConnectedLayer(int);
    ~FullyConnectedLayer();
    void feedForward();
    void setInput(double *);
};


#endif //INFERENCEENGINE_FULLYCONNECTEDLAYER_H
