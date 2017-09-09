//
// Created by Amr on 8/27/17.
//

#include <exception>
#include <numeric>
#include "Matrix.h"
#include "utils.h"
#include "old_vec/VecNN.h"


void Matrix::im2col(vector<int> &filterShape, int s, int pad, Matrix& out) {
    int index = 0;

    for (int i = 0; i < this->shape.at(1); i = i + s) { //length y
        for (int k = 0; k < this->shape.at(0); k = k + s) { //width x
            for (int j = 0; j < this->shape.at(2); j++) { // depth z
                int tt = k - pad + filterShape.at(0) - 1;
                int yy = i - pad + filterShape.at(0) - 1;
                if (tt < this->shape.at(0) + pad && yy < this->shape.at(0) + pad) {
                    for (int l = i - pad;
                         l < i - pad + filterShape.at(0) && l < this->shape.at(1) + pad; l++) { // for i
                        for (int m = k - pad;
                             m < k - pad + filterShape.at(0) && m < this->shape.at(0) + pad; m++) { //for j
                            int tem1 = m;
                            int tem2 = l;
                            if (tem1 < 0 || tem2 < 0 || tem1 >= this->shape.at(0) || tem2 >= shape.at(1)) {
                                index++;
                            } else {
//                                vector<int> in = {tem1, tem2, j};
                                //sdasdasll.push_back(in);
                                out.matrix.at(index) = this->matrix[tem1 + tem2*this->shape[0] + j*this->shape[0]*this->shape[1]];
                                //ll.push_back(this->at(in));
                                index++;
                            }

                        }
                    }
                }
            }
        }
    }
}

float Matrix::at(vector<int> index) {
    float out = calcuteOutput(index);
    if (out != -1)
        return matrix.at(out);
}

int Matrix::calcuteOutput(vector<int> &index) {
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

float Matrix::dotNoSSE(vector<float> &a, vector<float> &b) {
    float product = 0;
    for (int i = 0; i <= a.size()-1; i++)
            product = product + (a.at(i)*(b.at(i)));
    return product;
}

// HAVE TO CALL im2col before doing it.
void Matrix::dot(Matrix& filter, Matrix& out) {
    int x_dim = 0;
    int w_dim = 0;
    int index = 0;
    vector<float>::const_iterator first, last, first1, last1;
    vector<float> a, b;
//    __attribute__((aligned(sizeof(__m256)))) float aa[8],bb[8];
////    float aa[8], bb[8];
//    __m256 xx,yy;
    for (int i = 0; i < W_row_shape.at(0); i++) {
        first =filter.matrix.begin() + w_dim;
        last = filter.matrix.begin() + w_dim + W_row_shape.at(1);
        b = {first, last};
        //VecAVX w(W_row_shape.at(1), b);
        for (int j = 0; j < X_col_shape.at(1); j++) {
            first1 =this->matrix.begin() + x_dim;
            last1 = this->matrix.begin() + x_dim + X_col_shape.at(0);
            a = {first1, last1};
//            int size_mod_8 = a.size() % 8;
//            int number_of_iterations = std::ceil(a.size() / 8.0);
//            float sum = 0;
//            int size = 0;
//            for(int j = 0; j<number_of_iterations; j++){
//                for (int k = 0; k < 8; ++k) {
//                    if(size < a.size()){
//                        aa[k] = a[j*8 + k];
//                        bb[k] = b[j*8 + k];
//                    }
//                    else{
//                        aa[k] = 0;
//                        bb[k] = 0;
//                    }
//                    size++;
//                }
//
//                xx = _mm256_load_ps(aa);
//                yy = _mm256_load_ps(bb);
//                sum += dot_product( xx , yy);
//            }
            //float res = dotNoSSE(a, b); // can either do this
            float res = float(std::inner_product(a.begin(), a.end(), b.begin(), 0.0)); //or this

            out.matrix.at(index) = res;
            x_dim += X_col_shape.at(0);
            index++;
        }
        x_dim = 0;
        w_dim += W_row_shape.at(1);
    }
}

void Matrix::conv(Matrix& filter, int stride, int padding, Matrix& im, Matrix& out) {
    this->im2col(filter.shape, stride, padding, im);
    im.dot(filter, out);
}

// Should avoid returning by value here
Matrix Matrix::MaxRow(int kernel_size, int stride, int padding) {
    int pad = padding;
    int x = this->shape.at(0);
    x = x - kernel_size + 2 * pad;
    x = ceil(float(x) / float(stride));

    x = x + 1;
    int x_row = x * x;
    int depth = this->shape.at(2);
    vector<int> outSize = {x, x, depth};
    Matrix out(outSize);
    int index = 0;
    for (int j = 0; j < this->shape.at(2); j++) { // depth z
        for (int i = 0; i < this->shape.at(1); i = i + stride) { //length y
            for (int k = 0; k < this->shape.at(0); k = k + stride) { //width x
                int tt = k - pad + kernel_size - 1;
                int yy = i - pad + kernel_size - 1;
                if (tt < this->shape.at(0) + pad && yy < this->shape.at(0) + pad) {
                    float max = -(std::numeric_limits<float>::infinity());
                    for (int l = i - pad; l <= i - pad + kernel_size && l < this->shape.at(1) +
                                                                            pad; l++) { // for i int l = i-pad; l < i-pad+filterShape.at(0) && l<this->shape.at(1)+pad; l++
                        for (int m = k - pad; m <= k - pad + kernel_size && m < this->shape.at(0) + pad; m++) { //for j
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
    }
    return out;
}

Matrix Matrix::dotMM(Matrix &w) {
    vector<int> out_shape = {1, w.shape.at(1)};
    Matrix out(out_shape);
    int index = 0;
    vector<float>::const_iterator first =this->matrix.begin();
    vector<float>::const_iterator last = this->matrix.end();
    vector<float> b(first, last);
    vector<float>::const_iterator first1;
    vector<float>::const_iterator last1;
    vector<float> a;
    for (int i = 0; i < w.matrixSizeVector; i = i + w.shape.at(0)) {
        first1 =w.matrix.begin() + i;
        last1 = w.matrix.begin() + i + w.shape.at(0);
        a = {first1, last1};
        //out.matrix[index] = float(std::inner_product(a.begin(), a.end(), b.begin(), 0.0));
        out.matrix[index] = float(std::inner_product(a.begin(), a.end(), b.begin(), 0.0));//dotNoSSE(b, a);
        index++;
    }
    return out;
}

Matrix Matrix::transpose() {
    if (this->shape.size() != 2) {
        std::string str;
        for (int x : this->shape) {
            str += x + ", ";
        }
        throw std::logic_error("Cannot transpose matrix of shape [" + str + "], matrix must be 2D");
    }

    int N = this->shape[0];
    int M = this->shape[1];

    Matrix matrix_transposed({M, N});

    for (int n = 0; n < N * M; n++) {
        int i = n / N;
        int j = n % N;
        matrix_transposed.matrix[n] = this->matrix[M * j + i];
    }

    return matrix_transposed;
}

//Matrix Matrix::sub(Matrix &w) {
//    Matrix out(this->shape);
//    __attribute__((aligned (16))) float a[this->matrixSizeVector];
//    __attribute__((aligned (16))) float b[this->matrixSizeVector];
//    for (int k = 0; k < this->matrixSizeVector; k++) {
//        a[k] = this->matrix[k];
//        b[k] = w.matrix[k];
//    }
//
//    VecNN v(this->matrixSizeVector, a);
//    VecNN ww(w.matrixSizeVector, b);
//    out.matrix = v.sub(ww);
//    return out;
//}

void Matrix::addBiasNoSSE(Matrix &bias) {
    for (int i = 0; i<bias.shape.at(0); i++) {
        for (int j = 0 + i*this->shape.at(0)*this->shape.at(1); j < this->shape.at(0)*this->shape.at(1) + i*this->shape.at(0)*this->shape.at(1); j++) {
            this->matrix[j] = this->matrix[j] + bias.matrix[i];
        }
    }
}

void Matrix::subNoSSE(Matrix &m) {
    float average = 0;
    int j = 0;
    for (j = 0; j <m.matrix.size() ; ++j) {
        average += m.matrix[j];
    }
    average /= j;
    for(int i = 0; i< m.size(); i++){
        this->matrix[i] = this->matrix[i] - average;
    }
}

void Matrix::maxPooling(int kernel_size, int stride, int padding, Matrix& out) {
    int index = 0;
    for (int c = 0; c < this->shape.at(2); ++c) {
        for (int ph = 0; ph < out.shape.at(1); ++ph) {
            for (int pw = 0; pw < out.shape.at(0); ++pw) {
                int hstart = ph * stride - padding;
                int wstart = pw * stride - padding;
                int hend = min(hstart + kernel_size, this->shape.at(0));
                int wend = min(wstart + kernel_size, this->shape.at(1));
                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                const int pool_index = ph * out.shape.at(0) + pw + c*out.shape.at(0)*out.shape.at(1);
                for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                        const int index = h * this->shape.at(1) + w + c*this->shape.at(0)*this->shape.at(1);
                        if (this->matrix[index] > out.matrix[pool_index]) {
                            out.matrix[pool_index] = this->matrix[index];
                        }
                    }
                }
            }
        }
    }
}

void Matrix::im2col_cpu( Matrix* data_im, int pad_h, const int stride_h, Matrix* data_col, vector<int>& filterShape) {
    int index = 0;
    int output_h = ((data_im->shape[0] + (2 * pad_h) - filterShape[0]) / (stride_h)) + 1;
    const int channel_size = data_im->shape[0] * data_im->shape[1];
    //for (int channel = data_im->shape[2]; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < filterShape[0]; kernel_row++) {
            for (int kernel_col = 0; kernel_col < filterShape[0]; kernel_col++) {
                int input_row = -pad_h + kernel_row;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (!(0 < input_row && input_row< data_im->shape[0])) {
                        for (int output_cols = output_h; output_cols; output_cols--) {
                            data_col->matrix[index++] = 0;
                        }
                    } else {
                        int input_col = -pad_h + kernel_col;
                        for (int output_col = output_h; output_col; output_col--) {
                            if ((0< input_col && input_col< data_im->shape[1])) {
                                data_col->matrix[index++] = data_im->matrix[(input_row-pad_h) * data_im->shape[1] + (input_col-pad_h)];
                            } else {
                                data_col->matrix[index++] = 0;
                            }
                            input_col += stride_h;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    //}
    cout<<"works"<<endl;
}

