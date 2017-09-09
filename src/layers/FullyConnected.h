//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_FULLYCONNECTEDLAYER_H
#define INFERENCEENGINE_FULLYCONNECTEDLAYER_H

#include "abstract_layers/AbstractLayer.h"
#include "abstract_layers/AbstractWeightedLayer.h"

class FullyConnected : public AbstractWeightedLayer {
public:
    FullyConnected(string name, int num_of_outputs, std::unique_ptr<Matrix> weights, std::unique_ptr<Matrix> bias)
            : AbstractWeightedLayer(name, std::move(weights), std::move(bias), num_of_outputs) {};

    ~FullyConnected() {};

    void calculateOutput(Matrix &input_mat) {
        //vector<int> shape = this->weights->shape;
        //this->weights->shape = {input_mat.shape.at(0), input_mat.shape.at(1), input_mat.shape.at(2), this->weights->shape.at(1)};
        input_mat.conv(this->filter, 1, 0, this->im2col, this->output); //either do this
        //Matrix transposedW = this->weights->transpose();
        //input_mat = input_mat.dotMM(*this->weights);
//        input_mat = input_mat.dotMM(*this->weights); //or that
        output.addBiasNoSSE(*this->bias);
        //this->weights->shape = shape;
       // this->weights->conv(&input_mat,1,0);
    };

    void precompute(Matrix& in_mat){
        input = in_mat;
        filter = Matrix(weights->matrix, {in_mat.shape.at(0), in_mat.shape.at(1), in_mat.shape.at(2), this->weights->shape.at(1)});
        int x = input.shape.at(0);
        x = x - filter.shape[0] + 2 * 0;
        x = floor(float(x) / float(1));
        x = x + 1;
        int x_row = x * x;
        int x_col = filter.shape[0]*filter.shape[1]*filter.shape[2];
        vector<int> X_col_shape;
        vector<int> W_row_shape;
        X_col_shape.push_back(x_col);
        X_col_shape.push_back(x_row);
        int xx = filter.shape.at(filter.shape.size() - 1);
        W_row_shape.push_back(xx);
        W_row_shape.push_back(x_col);
        vector<int> im_shape = {X_col_shape.at(0), X_col_shape.at(1)};
        Matrix im(im_shape);
        im2col = im;
        im2col.W_row_shape = W_row_shape;
        im2col.X_col_shape = X_col_shape;
        vector<int> out_shape = {x, x, filter.shape.at(3)};
        Matrix out(out_shape);
        output = out;
    }

};


#endif //INFERENCEENGINE_FULLYCONNECTEDLAYER_H
