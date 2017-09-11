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
    for (int i = 0; i < W_row_shape.at(0); i++) {
        first =filter.matrix.begin() + w_dim;
        last = filter.matrix.begin() + w_dim + W_row_shape.at(1);
        b = {first, last};
        for (int j = 0; j < X_col_shape.at(1); j++) {
            first1 =this->matrix.begin() + x_dim;
            last1 = this->matrix.begin() + x_dim + X_col_shape.at(0);
            a = {first1, last1};
            float res = float(std::inner_product(a.begin(), a.end(), b.begin(), 0.0)); //or this
            //float res = dotNoSSE(b,a);
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
    //auto start = std::chrono::system_clock::now();
    //for (size_t counter = 0; counter < 10000; ++counter)
        im.dot(filter, out);

    //auto duration =
      //      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start) / 10000;
    //std::cout << "Completed function in " << duration.count() << " microseconds." << std::endl;

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