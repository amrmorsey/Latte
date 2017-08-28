//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_INPUTLAYER_H
#define INFERENCEENGINE_INPUTLAYER_H

#include "Layer.h"

class inputLayer: public Layer {
private:
    int no_of_input;
    int chanels;
    int height;
    int width;
    double *input;
    double *output;
    vector<int> input_dims;
    vector<int> output_dims;
public:
    inputLayer(int, int, int, int, int);
    ~inputLayer();
    void feedForward();
    void setInput(double *);
};


#endif //INFERENCEENGINE_INPUTLAYER_H
