//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_MAXPOOLINGLAYER_H
#define INFERENCEENGINE_MAXPOOLINGLAYER_H

#include "abstract_layers/AbstractLayer.h"

class MaxPool : public AbstractLayer {
private:
    int kernel_size;
    int stride;
    int padding;
public:
    MaxPool(string name, int kernel_size, int stride, int padding) : AbstractLayer(name), kernel_size(kernel_size),
                                                                     stride(stride), padding(padding) {};

    ~MaxPool() {};

    void calculateOutput(Matrix &input_mat) {
        input_mat.maxPooling(this->kernel_size, this->stride, this->padding, output);
    };

    void precompute(Matrix& in_mat){
        input = in_mat;
        int pad = padding;
        int x = in_mat.shape.at(0);
        x = x - kernel_size + 2 * pad;
        x = ceil(float(x) / float(stride));

        x = x + 1;
        int x_row = x * x;
        int depth = in_mat.shape.at(2);
        vector<int> outSize = {x, x, depth};
        Matrix out(outSize);
        output = out;
    }

};


#endif //INFERENCEENGINE_MAXPOOLINGLAYER_H
