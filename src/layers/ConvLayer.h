//
// Created by Amr on 8/21/17.
//

#ifndef INFERENCEENGINE_CONVLAYER_H
#define INFERENCEENGINE_CONVLAYER_H

#include <string>
#include <memory>
#include <chrono>
#include "abstract_layers/AbstractLayer.h"
#include "abstract_layers/AbstractWeightedLayer.h"

#include "../utils.h"

class ConvLayer : public AbstractWeightedLayer {

private:
    int kernel_size;
    int stride;
    int padding;

public:
    ConvLayer(std::string name, int num_of_outputs, std::unique_ptr<MatrixAVX> weights, std::unique_ptr<MatrixAVX> bias,
              int kernel_size, int stride, int padding) : AbstractWeightedLayer(name, std::move(weights),
                                                                                std::move(bias),
                                                                                num_of_outputs),
                                                          kernel_size(kernel_size), stride(stride),
                                                          padding(padding) {};

    ~ConvLayer() {};

    void calculateOutput(MatrixAVX &input_mat) {
        im2col(input_mat, weights->shape, im2col_out, stride, padding);
        std::vector<int> oldShape = weights.get()->shape;
        weights.get()->reshape({im2col_out.W_row_shape[1], im2col_out.W_row_shape[0]});
        im2col_out.dot_product(kept_dim, big_matrix_vec, big_reserve_size, s, chunk_range, output_before_bias);
        output_before_bias.add(biasMat, output);
        weights.get()->reshape(oldShape);
    };

    void precompute(std::vector<int>&in_mat) {
        int pad = padding;

        std::vector<int> filter_shape = this->weights.get()->shape;
        int x = in_mat[0];
        x = x - filter_shape[0] + 2 * pad;
        x = floor(float(x) / float(stride));
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
        std::vector<int> im_shape = {X_col_shape.at(0), X_col_shape.at(1)};
        MatrixAVX im(im_shape);
        im2col_out = im;
        im2col_out.X_col_shape = X_col_shape;
        im2col_out.W_row_shape = W_row_shape;
        std::vector<int> out_shape = {x, x, this->weights.get()->shape[3]};
        MatrixAVX out(out_shape);
        output_before_bias = out;
        output = out;


        for (int j = 0; j < output.shape.at(2); ++j) {
            for (int i = 0; i < output.shape.at(0) * output.shape.at(1); ++i) {
                biases.push_back(bias->getElement(j));
            }
        }
        biasMat = MatrixAVX(biases, output.shape);

        /////////////////////////////////////////////////////
        std::vector<int> oldShape = weights.get()->shape;
        weights.get()->reshape({im2col_out.W_row_shape[1], im2col_out.W_row_shape[0]});

        kept_dim = weights->shape[0];
        other_dim = weights->shape[1];
        repeated_dim = im2col_out.shape[1];
        smaller_mat = *weights.get();

        chunk_range = std::ceil(kept_dim / 8.0);
        big_reserve_size = chunk_range * 8 * repeated_dim;
        small_reserve_size = chunk_range * 8 * other_dim;

        small_matrix_vec = std::vector<float>(small_reserve_size, 0.0f);
        big_matrix_vec = std::vector<float>(big_reserve_size, 0.0f);

        unsigned int i = 0;
        unsigned int vec_index = 0;

        while (i < smaller_mat.size) {
            for (int j = 0; j < kept_dim; j++) {
                small_matrix_vec[vec_index + j] = smaller_mat.getElement(i + j);
            }
            vec_index += kept_dim;
            i += kept_dim;

            while (vec_index % 8 != 0)
                ++vec_index;
        }

        MatrixAVX small(small_matrix_vec, {(int)small_reserve_size, 1});

        s = small;
        weights.get()->reshape(oldShape);

        MatrixAVX mat(big_matrix_vec, {(int)big_reserve_size, 1});
        im2col_out = mat;
        im2col_out.X_col_shape = X_col_shape;
        im2col_out.W_row_shape = W_row_shape;
    }
};


#endif //INFERENCEENGINE_CONVLAYER_H
