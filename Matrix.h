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

using namespace std;

class Matrix {
private:
//    unique_ptr<vector<int>> shape;
//    unique_ptr<vector<float>> matrix;
    vector<int> shape;
    vector<float> matrix;
    int matrixSizeVector;
    int calcuteOutput(vector<int> index);
    vector<int> X_col_shape;
    vector<int> W_row_shape;
public:
    Matrix(vector<float> m, vector<int> s);

    Matrix(vector<int> s);

    ~Matrix() {};

    Matrix im2col(vector<int>, int s);

    float at(vector<int>);

    void set(vector<int>, float);

    vector<int> calculateIndex(int x);

    Matrix dot(Matrix);

    Matrix conv(Matrix, int);
};


#endif //INFERENCEENGINE_MATRIX_H
