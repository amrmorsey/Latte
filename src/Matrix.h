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
#include "VecNN.h"
#include "VecAVX.h"
#include <limits>
#include <cmath>
#include <mm_malloc.h>

using namespace std;

class Matrix {
private:
//    unique_ptr<vector<int>> shape;
//    unique_ptr<vector<float>> matrix;
    vector<int> shape;
    vector<float> matrix;
    int matrixSizeVector;
    vector<int> X_col_shape;
    vector<int> W_row_shape;

    Matrix im2col(vector<int>&, int s, int, int);

    int calcuteOutput(vector<int> &index);

public:
    Matrix();

    explicit Matrix(vector<float> m, vector<int> s);

    explicit Matrix(vector<int> s);

    ~Matrix() {};

    float at(vector<int>);

    void set(vector<int>, float);

    vector<int> calculateIndex(int x);

    Matrix dot(Matrix, int);

    Matrix conv(Matrix&, int, bool);

    Matrix MaxRow(Matrix&, int, bool);

    void relU();

    void sigmoidApp();

    vector<float> softmax(float& temp);
};


#endif //INFERENCEENGINE_MATRIX_H
