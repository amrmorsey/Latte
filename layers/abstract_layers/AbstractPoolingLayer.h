//
// Created by shadyf on 29/08/17.
//

#ifndef INFERENCEENGINE_POOLING_LAYER_H
#define INFERENCEENGINE_POOLING_LAYER_H

#include <string>
#include <vector>

class AbstractPoolingLayer {
private:
    int kernel_size;
    int stride;
    double *input;
    double *output;
    std::vector<int> input_dims;
    std::vector<int> output_dims;
public:
    std::string name;

    AbstractPoolingLayer() {};

    ~AbstractPoolingLayer() {};

    void feedForward();

    void setInput(double *);
};


#endif //INFERENCEENGINE_POOLING_LAYER_H
