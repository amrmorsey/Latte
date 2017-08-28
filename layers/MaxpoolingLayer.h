//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_MAXPOOLINGLAYER_H
#define INFERENCEENGINE_MAXPOOLINGLAYER_H

#include "Layer.h"

class MaxpoolingLayer: public Layer{
private:
    int kernel_size;
    int stride;
    double *input;
    double *output;
    vector<int> input_dims;
    vector<int> output_dims;
public:
    MaxpoolingLayer(int, int);
    ~MaxpoolingLayer();
    void feedForward();
    void setInput(double *);
};


#endif //INFERENCEENGINE_MAXPOOLINGLAYER_H
