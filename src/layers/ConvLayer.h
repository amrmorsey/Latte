//
// Created by Amr on 8/21/17.
//

#ifndef INFERENCEENGINE_CONVLAYER_H
#define INFERENCEENGINE_CONVLAYER_H

#include <string>
#include <memory>
#include "abstract_layers/AbstractLayer.h"
#include "abstract_layers/AbstractWeightedLayer.h"

class ConvLayer : public AbstractWeightedLayer {

private:
    int kernel_size;
    int stride;
    int padding;
public:
    ConvLayer(std::string name, int num_of_outputs, std::unique_ptr<Matrix> weights, std::unique_ptr<Matrix> bias,
              int kernel_size, int stride, int padding) : AbstractWeightedLayer(name, std::move(weights),
                                                                                std::move(bias),
                                                                                num_of_outputs),
                                                          kernel_size(kernel_size), stride(stride),
                                                          padding(padding) {};

    ~ConvLayer() {};

    void calculateOutput(Matrix &input_mat) {
        input_mat.conv(*this->weights, this->stride, this->padding, this->im2col, this->output);
        output.addBiasNoSSE(*this->bias);
    };

    void precompute(Matrix& in_mat){
        input = in_mat;
        int x = input.shape.at(0);
        x = x - kernel_size + 2 * padding;
        x = floor(float(x) / float(stride));
        x = x + 1;
        int x_row = x * x;
        int x_col = weights->shape[0]*weights->shape[1]*weights->shape[2];
        vector<int> X_col_shape;
        vector<int> W_row_shape;
        X_col_shape.push_back(x_col);
        X_col_shape.push_back(x_row);
        int xx = weights->shape.at(weights->shape.size() - 1);
        W_row_shape.push_back(xx);
        W_row_shape.push_back(x_col);
        vector<int> im_shape = {X_col_shape.at(0), X_col_shape.at(1)};
        Matrix im(im_shape);
        im2col = im;
        im2col.W_row_shape = W_row_shape;
        im2col.X_col_shape = X_col_shape;
        vector<int> out_shape = {x, x, weights->shape.at(3)};
        Matrix out(out_shape);
        output = out;
    }
};


#endif //INFERENCEENGINE_CONVLAYER_H
