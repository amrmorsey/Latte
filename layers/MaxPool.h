//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_MAXPOOLINGLAYER_H
#define INFERENCEENGINE_MAXPOOLINGLAYER_H

#include "abstract_layers/AbstractLayer.h"

class MaxPoolingLayer: public AbstractLayer{
private:
    int kernel_size;
    int stride;
    double *input;
    double *output;
    vector<int> input_dims;
    vector<int> output_dims;
public:
    MaxPoolingLayer(int, int);
    ~MaxPoolingLayer();
    void feedForward();
    void setInput(double *);
};


#endif //INFERENCEENGINE_MAXPOOLINGLAYER_H
