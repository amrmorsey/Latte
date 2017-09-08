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
        int index = 0;

        // for (int n = 0; n < bottom[0]->num(); ++n) {
        for (int c = 0; c < input_mat.shape[2]; ++c) {
            for (int ph = 0; ph < out.shape[1]; ++ph) {
                for (int pw = 0; pw < out.shape[0]; ++pw) {
                    int hstart = ph * stride - padding;
                    int wstart = pw * stride - padding;
                    int hend = std::min(hstart + kernel_size, input_mat.shape[0]);
                    int wend = std::min(wstart + kernel_size, input_mat.shape[1]);
                    hstart = std::max(hstart, 0);
                    wstart = std::max(wstart, 0);
                    const int pool_index = ph * out.shape[0] + pw + c*out.shape[0]*out.shape[1];
                    for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                            const int index = h * input_mat.shape[1] + w + c*input_mat.shape[0]*input_mat.shape[1];
                            if (input_mat.getElement(index) > out.getElement(pool_index)) {
                                out.setElement(pool_index, input_mat.getElement(index));
                            }
                        }
                    }
                }
            }
        }
    }
    void calculateOutput(MatrixAVX &input_mat) {
        int x = input_mat.shape[0];
        x = x - kernel_size + 2 * padding;

        x = std::ceil(float(x) / float(stride));

        x = x + 1;
        int x_row = x * x;
        int depth = input_mat.shape[2];
        std::vector<int> outSize = {x, x, depth};
        MatrixAVX out(outSize);

        maxPool(input_mat,out);

        input_mat = out;
    };
};


#endif //INFERENCEENGINE_MAXPOOLINGLAYER_H
