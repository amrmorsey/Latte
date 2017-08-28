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
using namespace std;

class Matrix {
private:
//    unique_ptr<vector<int>> shape;
//    unique_ptr<vector<float>> matrix;
    vector<int> shape;
    vector<float> matrix;
    vector<int> calcuteOutput();
public:
    Matrix(vector<int> s, vector<float> m) : matrix(m), shape(s){};
    ~Matrix(){};
    Matrix im2col(vector<int>, int s, int f);
    float at(vector<int>);
};


#endif //INFERENCEENGINE_MATRIX_H
