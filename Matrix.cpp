//
// Created by Amr on 8/27/17.
//

#include "Matrix.h"


Matrix Matrix::im2col(vector<int> filterShape, int s, int f) {
    vector<int> X_col_shape;
    vector<int> W_row_shape;
    int pad = filterShape.at(1);
    pad = pad -1;
    pad = pad/2;
    int x = this->shape.at(1);
    x = x - filterShape.at(1) + pad;
    x = x/s;
    x = x +1;
    int x_row = x*x;
    int x_col = 1;
    for(int i = 0; i<filterShape.size(); i++){
        x_col *= filterShape.at(i);
    }
    X_col_shape.push_back(x_col);
    X_col_shape.push_back(x_row);

    W_row_shape.push_back(f);
    W_row_shape.push_back(x_col);

    cout<<"So far so good!"<<endl;




}



//void im2col_cpu(const Dtype* data_im, const int channels,
//                const int height, const int width, const int kernel_h, const int kernel_w,
//                const int pad_h, const int pad_w,
//                const int stride_h, const int stride_w,
//                const int dilation_h, const int dilation_w,
//                Dtype* data_col) {
//    const int output_h = (height + 2 * pad_h -
//                          (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
//    const int output_w = (width + 2 * pad_w -
//                          (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
//    const int channel_size = height * width;
//    for (int channel = channels; channel--; data_im += channel_size) {
//        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
//            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
//                int input_row = -pad_h + kernel_row * dilation_h;
//                for (int output_rows = output_h; output_rows; output_rows--) {
//                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
//                        for (int output_cols = output_w; output_cols; output_cols--) {
//                            *(data_col++) = 0;
//                        }
//                    } else {
//                        int input_col = -pad_w + kernel_col * dilation_w;
//                        for (int output_col = output_w; output_col; output_col--) {
//                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
//                                *(data_col++) = data_im[input_row * width + input_col];
//                            } else {
//                                *(data_col++) = 0;
//                            }
//                            input_col += stride_w;
//                        }
//                    }
//                    input_row += stride_h;
//                }
//            }
//        }
//    }
//}
//
//inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
//    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
//}
