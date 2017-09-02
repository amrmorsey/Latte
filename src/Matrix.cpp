//
// Created by Amr on 8/27/17.
//

#include "Matrix.h"


Matrix Matrix::im2col(vector<int> filterShape, int s, int pad, int x) {
    int x_row = x * x;
    int x_col = 1;
    for (int i = 0; i < filterShape.size() - 1; i++) {
        x_col *= filterShape.at(i);
    }
    X_col_shape.push_back(x_col);
    X_col_shape.push_back(x_row);
    int xx = filterShape.at(filterShape.size() - 1);
    W_row_shape.push_back(xx);
    W_row_shape.push_back(x_col);

    //(x, y, z) = Z*(Dim_Y*Dim_X) + y*DIM_X + x
    vector<int> out_shape = {X_col_shape.at(0), X_col_shape.at(1)};
    Matrix out(out_shape);
    int index = 0;

    for (int i = 0; i < this->shape.at(1); i = i + s) { //length y
        for (int k = 0; k < this->shape.at(0); k = k + s) { //width x
            for (int j = 0; j < this->shape.at(2); j++) { // depth z
                for (int l = i-pad; l < i-pad+filterShape.at(0); l++) { // for i
                    for (int m = k-pad; m < k-pad+filterShape.at(0); m++) { //for j
                        int tem1 = m;
                        int tem2 = l;
                        if (tem1 < 0 || tem2 < 0 || tem1 >= this->shape.at(0) || tem2 >= shape.at(1)) {
                            index++;
                        } else {
                            vector<int> in = {tem1, tem2, j};
                            out.matrix.at(index) = this->at(in);
                            index++;
                        }

                    }
                }
            }
        }
    }
    out.W_row_shape = this->W_row_shape;
    out.X_col_shape = this->X_col_shape;
    return out;
}

float Matrix::at(vector<int> index) {
    float out = calcuteOutput(index);
    if (out != -1)
        return matrix.at(out);
}

int Matrix::calcuteOutput(vector<int> index) {
    int out = 0;
    for (int i = 0; i < this->shape.size(); i++) {
        int x = index.at(i);
        int y = 1;
        for (int j = i - 1; j >= 0; j--) {
            y *= this->shape.at(j);
        }
        out += x * y;
    }

    if (out >= 0 && out < matrixSizeVector)
        return out;
    else
        return -1;
}

void Matrix::set(vector<int> index, float value) {
    int out = calcuteOutput(index);
    if (out != -1) {
        this->matrix.at(out) = value;
    }
}

vector<int> Matrix::calculateIndex(int x) {
    vector<int> index;
    int temp = 0;
    for (int i = this->shape.size() - 1; i >= 0; i--) {
        temp = x;
        int y = 1;
        for (int j = 0; j < i; j++) {
            y *= this->shape.at(j);
        }
        int t = temp / y;
        index.push_back(t);
        x -= (t * y);
    }
    std::reverse(index.begin(), index.end());
    return index;
}

Matrix::Matrix(vector<int> s) {
    shape = s;
    int x = 1;
    for (int i = 0; i < s.size(); ++i) {
        x *= s.at(i);
    }
    matrix.resize(x);
    matrixSizeVector = x;
}

Matrix::Matrix(vector<float> m, vector<int> s) : matrix(m), shape(s) {
    matrixSizeVector = m.size();
}

Matrix::Matrix() {
    shape = {0};
    matrixSizeVector = 0;
}

// HAVE TO CALL im2col before doing it.
Matrix Matrix::dot(Matrix filter, int x) {
    vector<int> out_shape = {x, x, filter.shape.at(3)};
    Matrix out(out_shape);
    int x_dim = 0;
    int w_dim = 0;
    int index = 0;
    for (int i = 0; i < W_row_shape.at(0); i++) {
        //__attribute__((aligned (32))) float b[W_row_shape.at(1)];
        __attribute__((aligned (16))) float b[W_row_shape.at(1)];
        //float *b = (float *)_mm_malloc(W_row_shape.at(1)*sizeof(float), 32);
        for (int k = 0; k < W_row_shape.at(1); k++) {
            b[k] = filter.matrix[k + w_dim];
        }
        VecNN w(W_row_shape.at(1), b);
        //VecAVX w(W_row_shape.at(1), b);
        for (int j = 0; j < X_col_shape.at(1); j++) {
           // __attribute__((aligned (32))) float a[X_col_shape.at(0)];// = &this->matrix[x_dim];
            __attribute__((aligned (16))) float a[X_col_shape.at(0)];
            //float *a = (float *)_mm_malloc(X_col_shape.at(0)*sizeof(float), 32);
            for (int k = 0; k < X_col_shape.at(0); k++) {
                a[k] = this->matrix[k + x_dim];
            }
            VecNN v(X_col_shape.at(0), a);
            //VecAVX v(X_col_shape.at(0), a);
            float res = v.dot(w);
            out.matrix.at(index) = res;
            x_dim += X_col_shape.at(0);
            index++;
            //_mm_free(a);
        }

       // _mm_free(b);
        x_dim = 0;
        w_dim += W_row_shape.at(1);
    }

    return out;
}

Matrix Matrix::conv(Matrix filter, int s, bool padding) {
    int pad = 0;
    if (padding) {
        pad = filter.shape.at(0);
        pad = pad - 1;
        pad = pad / 2;
    } else {
        pad = 0;
    }

    int x = this->shape.at(0);
    x = x - filter.shape.at(0) + 2 * pad;
    x = x / s;
    x = x + 1;
    Matrix out = this->im2col(filter.shape, s, pad, x);
    return out.dot(filter,x);
}

Matrix Matrix::MaxRow(Matrix filter, int s, bool padding) {
    int pad = 0;
    int filter_width = filter.shape.at(0) / 2;
    if (padding) {
        pad = filter.shape.at(0);
        pad = pad - 1;
        pad = pad / 2;
    } else {
        pad = 0;
    }
    int x = this->shape.at(0);
    x = x - filter.shape.at(0) + 2 * pad;
    x = x / s;
    x = x + 1;
    int x_row = x * x;
    int depth = this->shape.at(2);
    vector<int> outSize = {x, x, depth};
    Matrix out(outSize);
    int index = 0;

    for (int j = 0; j < this->shape.at(2); j++) { // depth z
        for (int i = 0; i < this->shape.at(1); i = i + s) { //length y
            for (int k = 0; k < this->shape.at(0); k = k + s) { //width x
                float max = -(std::numeric_limits<float>::infinity());
                for (int l = i-pad; l <= i-pad + filter.shape.at(0); l++) { // for i
                    for (int m = k-pad; m <= k-pad+shape.at(0); m++) { //for j
                        int tem1 = m;
                        int tem2 = l;
                        if (tem1 < 0 || tem2 < 0 || tem1 >= this->shape.at(0) || tem2 >= shape.at(1)) {
                            if (0 > max)
                                max = 0;
                        } else {
                            vector<int> in = {tem1, tem2, j};
                            if (this->at(in) > max)
                                max = this->at(in);
                        }
                    }
                }
                out.matrix.at(index) = max;
                index++;
            }
        }
    }
    return out;
}

void Matrix::relU() {
    for(int i = 0; i<this->matrix.size(); i++){
        if(this->matrix[i] < 0)
            this->matrix[i] = 0;
    }
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
