//
// Created by Amr on 8/27/17.
//

#ifndef INFERENCEENGINE_MATRIX_H
#define INFERENCEENGINE_MATRIX_H

#include <immintrin.h>
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <mm_malloc.h>

using namespace std;

class Matrix {
private:
    void im2col(vector<int>&, int s, int, Matrix&);

    //Matrix dot(Matrix*, int);
public:
    vector<int> X_col_shape;
    vector<int> W_row_shape;
    int calcuteOutput(vector<int> &index);
    int matrixSizeVector;
    // matrix should be a private membder
    vector<int> shape;
    vector<float> matrix;

    Matrix();

    explicit Matrix(vector<float> m, vector<int> s);

    explicit Matrix(vector<int> s);

    ~Matrix() {};

    float at(vector<int>);

    void set(vector<int>, float);

    vector<int> calculateIndex(int x);


    void conv(Matrix& filter, int s, int padding, Matrix& im, Matrix& out);

    Matrix MaxRow(int kernel_size, int stride, int padding);

    Matrix transpose();

    unsigned long size() {
        return this->matrix.size();
    }

    Matrix dotMM(Matrix&);

    Matrix sub(Matrix&);

    void addBiasNoSSE(Matrix&);

    void subNoSSE(Matrix&);

    float dotNoSSE(vector<float> &a, vector<float> &b);

    void maxPooling(int, int, int, Matrix&);

    void im2col_cpu(Matrix *data_im, int pad_h, const int stride_h, Matrix *data_col, vector<int> &filterShape);

    void dot(Matrix &a, Matrix &out);

};


#endif //INFERENCEENGINE_MATRIX_H
