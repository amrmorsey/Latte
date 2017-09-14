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
// Calculates the output of the FullyConnected.
    void calculateOutput(MatrixAVX &input_mat) {
        im2col(input_mat, filter.shape, im2col_out, 1, 0);
        weights.get()->reshape({im2col_out.W_row_shape[1], im2col_out.W_row_shape[0]});
        im2col_out.reshape({im2col_out.X_col_shape[1], im2col_out.X_col_shape[0]});
        im2col_out.dot_product(kept_dim, big_matrix_vec, big_reserve_size, s, chunk_range, output_before_bias);
        output_before_bias.add(biasMat, output);
    };
// Sets up the FullyConnected layer, it takes the shape of the matrix before it to compute its own matrices.
    void precompute(std::vector<int> &in_mat) {
        input = MatrixAVX(in_mat);

        // Compute the size output of the im2col function, the X col, and W row matrices.
        filter = MatrixAVX(weights->xmm, {in_mat[0], in_mat[1], in_mat[2], this->weights->shape[1]});
        int x = input.shape[0];
        x = x - filter.shape[0];
        x = floor(float(x) / float(1));
        x = x + 1;
        int x_row = x * x;
        int x_col = filter.shape[0] * filter.shape[1] * filter.shape[2];
        std::vector<int> X_col_shape;
        std::vector<int> W_row_shape;
        X_col_shape.push_back(x_col);
        X_col_shape.push_back(x_row);
        int xx = filter.shape.at(filter.shape.size() - 1);
        W_row_shape.push_back(xx);
        W_row_shape.push_back(x_col);
        std::vector<int> im_shape = {X_col_shape.at(0), X_col_shape.at(1)};
        MatrixAVX im(im_shape);
        im2col_out = im;
        im2col_out.W_row_shape = W_row_shape;
        im2col_out.X_col_shape = X_col_shape;

        // Computes the output size of the matrix of the dot product function and of the add bias.
        std::vector<int> out_shape = {x, x, filter.shape.at(3)};
        MatrixAVX out(out_shape);
        output_before_bias = out;
        output = out;

        // Set up the bias matrix so that its ready to be added to the output after the dot product.
        for (int j = 0; j < output.shape.at(2); ++j) {
            for (int i = 0; i < output.shape.at(0) * output.shape.at(1); ++i) {
                biases.push_back(bias->getElement(j));
            }
        }
        biasMat = MatrixAVX(biases, output.shape);

        // Setting up the weights matrix and reshaping it to the desired shape for the dot product.
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

        MatrixAVX small(small_matrix_vec, {(int) small_reserve_size, 1});
        s = small;
        weights.get()->reshape(oldShape);
    }
};


#endif //INFERENCEENGINE_FULLYCONNECTEDLAYER_H
