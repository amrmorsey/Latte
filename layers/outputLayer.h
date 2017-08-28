//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_OUTPUTLAYER_H
#define INFERENCEENGINE_OUTPUTLAYER_H

#include "Layer.h"

class outputLayer: public Layer {
private:
    int output_size;
    double *input;
    double *output;
    vector<int> input_dims;
    vector<int> output_dims;
public:
    outputLayer();
    ~outputLayer();
    void feedForward();
    void setInput(double *);
};


#endif //INFERENCEENGINE_OUTPUTLAYER_H
