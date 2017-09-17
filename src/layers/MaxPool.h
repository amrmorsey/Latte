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
    MaxPool(std::string name, int kernel_size, int stride, int padding) : AbstractLayer(name), kernel_size(kernel_size),
                                                                          stride(stride), padding(padding) {};

    ~MaxPool() {};
    void maxPool(MatrixAVX &input_mat, MatrixAVX &out) {
        // for (int n = 0; n < bottom[0]->num(); ++n) {
        float temp;
        for (int c = 0; c < input_mat.shape[2]; ++c) {
            for (int ph = 0; ph < out.shape[1]; ++ph) {
                for (int pw = 0; pw < out.shape[0]; ++pw) {
                    int hstart = ph * stride - padding;
                    int wstart = pw * stride - padding;
                    int hend = std::min(hstart + kernel_size, input_mat.shape[0]);
                    int wend = std::min(wstart + kernel_size, input_mat.shape[1]);
                    hstart = std::max(hstart, 0);
                    wstart = std::max(wstart, 0);
                    const int pool_index = ph * out.shape[0] + pw + c * out.shape[0] * out.shape[1];
                    temp = 0;
                    for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                            const int index = h * input_mat.shape[1] + w + c * input_mat.shape[0] * input_mat.shape[1];
                            temp = std::max(temp, input_mat.getElement(index));
                        }
                    }
                    out.setElement(pool_index, temp);
                }
            }
        }
    }
// Calculates the output of the Maxpooling.
    void calculateOutput(MatrixAVX &input_mat) {
        maxPool(input_mat, output);
    };
// Sets up the Maxpooling layer, it takes the shape of the matrix before it to compute its own matrices.
    void precompute(std::vector<int> &in_mat) {
        //Calculate output size.
        int x = in_mat[0];
        x = x - kernel_size + 2 * padding;

        x = std::ceil(float(x) / float(stride));

        x = x + 1;
        int depth = in_mat[2];
        std::vector<int> outSize = {x, x, depth};
        MatrixAVX out(outSize);
        output = out;
    }
};


#endif //INFERENCEENGINE_MAXPOOLINGLAYER_H
