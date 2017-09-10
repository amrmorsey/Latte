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
        im2col(input_mat, weights.get()->shape, im2col_out, 1, 0);
        weights.get()->reshape({im2col_out.W_row_shape[1], im2col_out.W_row_shape[0]});
        im2col_out.reshape({im2col_out.X_col_shape[1], im2col_out.X_col_shape[0]});
        im2col_out.dot_product(*weights.get(), output_before_bias);
        output_before_bias.add(biases, biases_stranglers, output);//asdasd
    };

    void precompute(MatrixAVX& in_mat){
        input = in_mat;
        filter = MatrixAVX(weights->xmm, {in_mat.shape[0], in_mat.shape[1], in_mat.shape[2], this->weights->shape[1]});
        int x = input.shape[0];
        x = x - filter.shape[0];
        x = floor(float(x)/float(1));
        x = x+1;
        int x_row = x*x;
        int x_col = filter.shape[0]*filter.shape[1]*filter.shape[2];
        std::vector<int> X_col_shape;
        std::vector<int> W_row_shape;
        X_col_shape.push_back(x_col);
        X_col_shape.push_back(x_row);
        int xx = filter.shape.at(filter.shape.size()-1);
        W_row_shape.push_back(xx);
        W_row_shape.push_back(x_col);
        std::vector<int> im_shape = {X_col_shape.at(0), X_col_shape.at(1)};
        MatrixAVX im(im_shape);
        im2col_out = im;
        im2col_out.W_row_shape = W_row_shape;
        im2col_out.X_col_shape = X_col_shape;
        std::vector<int> out_shape = {x,x, filter.shape.at(3)};
        MatrixAVX out(out_shape);
        output_before_bias = out;
        output = out;

        int rem = (output_before_bias.shape[0] * output_before_bias.shape[0]) % 8;
        for (int i = 0; i < bias.get()->size; i++)
            biases.push_back(_mm256_set1_ps(bias.get()->xmm[0][i]));

        if (rem) {
            __m256i mask = _mm256_setr_epi32(-rem, 1 - rem, 2 - rem, 3 - rem, 4 - rem, 5 - rem, 6 - rem, 7 - rem);
            __m256i mask2 = _mm256_setr_epi32(7 - rem, 6 - rem, 5 - rem, 4 - rem, 3 - rem, 2 - rem, 1 - rem, -rem);

            for (int k = 0; k < biases.size(); ++k) {
                if (k + 1 < biases.size()) {
                    __m256 x = _mm256_add_ps(_mm256_maskload_ps(reinterpret_cast<const float *>(&biases[k]), mask),
                                             _mm256_maskload_ps(reinterpret_cast<const float *>(&biases[k + 1]),
                                                                mask2));
                    biases_stranglers.push_back(x);
                } else
                    biases_stranglers.push_back(_mm256_maskload_ps(reinterpret_cast<const float *>(&biases[k]), mask));
            }
        }
    }
};


#endif //INFERENCEENGINE_FULLYCONNECTEDLAYER_H
