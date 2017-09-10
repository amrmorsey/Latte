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


        im2col(input_mat, weights.get()->shape, im2col_out, stride, padding);
        weights.get()->reshape({im2col_out.W_row_shape[1], im2col_out.W_row_shape[0]});
        im2col_out.reshape({im2col_out.X_col_shape[1], im2col_out.X_col_shape[0]});
        auto start = std::chrono::system_clock::now();
        for (size_t counter = 0; counter < 10000; ++counter) {
            im2col_out.dot_product(*weights.get(), output_before_bias);
        }
        auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start) / 10000;
        output_before_bias.add(biases, biases_stranglers, output);
        std::cout << "Completed function in " << duration.count() << " microseconds." << std::endl;
    };

    void precompute(MatrixAVX &in_mat) {
        int pad = padding;

        std::vector<int> filter_shape = this->weights.get()->shape;
        int x = in_mat.shape[0];
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

        //int limit = std::floor((output_before_bias.shape[0] * output_before_bias.shape[1])/8.0);
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


#endif //INFERENCEENGINE_CONVLAYER_H
