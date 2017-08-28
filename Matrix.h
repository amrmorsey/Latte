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
using namespace std;

class Matrix {
private:
//    unique_ptr<vector<int>> shape;
//    unique_ptr<vector<float>> matrix;
    vector<int> shape;
    vector<float> matrix;
    int calcuteOutput(vector<int> index);
public:
    Matrix(vector<int> s, vector<float> m) : matrix(m), shape(s){};
    Matrix(vector<int> s): shape(s){};
    ~Matrix(){};
    Matrix im2col(vector<int>, int s);
    float at(vector<int>);
    void set(vector<int>, float);
    vector<int> calculateIndex(int x);
};


#endif //INFERENCEENGINE_MATRIX_H
