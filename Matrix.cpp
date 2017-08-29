//
// Created by Amr on 8/27/17.
//

#include "Matrix.h"


Matrix Matrix::im2col(vector<int> filterShape, int s) {
    vector<int> X_col_shape;
    vector<int> W_row_shape;
    int pad = filterShape.at(0);
    pad = pad -1;
    pad = pad/2;
    int x = this->shape.at(0);
    x = x - filterShape.at(0) + pad;
    x = x/s;
    x = x +1;
    int x_row = x*x;
    int x_col = 1;
    for(int i = 0; i<filterShape.size()-1; i++){
        x_col *= filterShape.at(i);
    }
    X_col_shape.push_back(x_col);
    X_col_shape.push_back(x_row);
    int xx = filterShape.at(filterShape.size()-1);
    W_row_shape.push_back(xx);
    W_row_shape.push_back(x_col);

    cout<<"So far so good!"<<endl;

    //(x, y, z) = Z*(Dim_Y*Dim_X) + y*DIM_X + x
    vector<int> out_shape = {W_row_shape.at(1), X_col_shape.at(1)};
    Matrix out(out_shape);

//    for (int i = 0; i < this->matrix.size(); i = i+s) {
//        vector<int> in = this->calculateIndex(i);
//
//        for (int j = 0; j < in.size(); j++) {
//
//        }
//    }

    for (int i = 0; i < this->shape.at(1); i= i+s) { //length y
        for (int j = 0; j <this->shape.at(2) ; j++) { // depth z
            for (int k = 0; k < this->shape.at(0); k= k+s) { //width x
                for (int l = -pad; l <=pad ; l++) { // for j
                    for (int m = -pad; m <=pad ; m++) {
                        int tem1 = k+m;
                        int tem2 = i+l;
                        if(tem1 <0 || tem2<0){
                            out.matrix.push_back(0);
                        }else{
                            vector<int> in ={tem1, tem2, j};
                            out.matrix.push_back(this->at(in));
                        }

                    }
                }
            }
        }
    }

    return out;
}

float Matrix::at(vector<int> index) {
    float out = calcuteOutput(index);
    if(out != -1)
        return matrix.at(out);
}

int Matrix::calcuteOutput(vector<int> index) {
    int out = 0;
    for (int i = 0; i <this->shape.size() ; i++) {
        out += index.at(i);
        for (int j = i-1; j >=0 ; j--) {
            out *= this->shape.at(j);
        }
    }

    if(out<this->shape.size())
        return out;
    else
        return -1;
}

void Matrix::set(vector<int> index, float value) {
    int out = calcuteOutput(index);
    if(out != -1){
        this->matrix.at(out) = value;
    }
}

vector<int> Matrix::calculateIndex(int x) {
    vector<int> index;
    int temp = 0;
    for (int i = this->shape.size()-1; i >=0; i--) {
        temp = x;
        int y = 1;
        for (int j = 0; j <i ; j++) {
            y *= this->shape.at(j);
        }
        int t = temp/y;
        index.push_back(t);
        x -= (t*y);
    }
    std::reverse(index.begin(), index.end());
    return index;
}

Matrix::Matrix(vector<int> s) {
    shape = s;
    int x = 1;
    for (int i = 0; i < s.size(); ++i) {
        x *=s.at(i);
    }
    matrix.resize(x);
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
