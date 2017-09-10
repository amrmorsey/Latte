//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_FULLYCONNECTEDLAYER_H
#define INFERENCEENGINE_FULLYCONNECTEDLAYER_H

#include "abstract_layers/AbstractLayer.h"
#include "abstract_layers/AbstractWeightedLayer.h"

class FullyConnected : public AbstractWeightedLayer {
public:
    FullyConnected(std::string name, int num_of_outputs, std::unique_ptr<MatrixAVX> weights,
                   std::unique_ptr<MatrixAVX> bias)
            : AbstractWeightedLayer(name, std::move(weights), std::move(bias), num_of_outputs) {};

    ~FullyConnected() {};

    void calculateOutput(MatrixAVX &input_mat) {
        std::vector<int> shape = this->weights->shape;
        this->weights->shape = {input_mat.shape.at(0), input_mat.shape.at(1), input_mat.shape.at(2),
                                this->weights->shape.at(1)};

        int pad = 0;
        int stride = 1;

        std::vector<int> filter_shape = this->weights.get()->shape;
        int x = input_mat.shape[0];
        x = x - filter_shape[0] + 2 * pad;
        x = std::floor(float(x) / float(stride));
        x = x + 1;

        int x_row = x * x;
        int x_col = 1;
        std::vector<int> X_col_shape;
        std::vector<int> W_row_shape;
        for (int i = 0; i < filter_shape.size() - 1; i++) {
            x_col *= filter_shape[i];
        }
        X_col_shape.push_back(x_col);
        X_col_shape.push_back(x_row);
        int xx = filter_shape.at(filter_shape.size() - 1);
        W_row_shape.push_back(xx);
        W_row_shape.push_back(x_col);

        //(x, y, z) = Z*(Dim_Y*Dim_X) + y*DIM_X + x
        std::vector<int> out_shape = {X_col_shape.at(0), X_col_shape.at(1)};
        MatrixAVX out(out_shape), out_dot(out_shape), out_bias(out_shape);

        im2col(input_mat, weights.get()->shape, out, stride, pad);
        out.dot_product(*weights.get(), out_dot);
        out_dot.add(biases,biases_stranglers, out_bias);
        input_mat = out_bias;
    };

    void precompute(MatrixAVX& in_mat){

    }
};


#endif //INFERENCEENGINE_FULLYCONNECTEDLAYER_H
