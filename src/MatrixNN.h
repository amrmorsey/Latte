//
// Created by Amr on 8/23/17.
//

#ifndef INFERENCEENGINE_MATRIXNN_H
#define INFERENCEENGINE_MATRIXNN_H

#include "VecNN.h"


class MatrixNN {
private:
    VecNN *matrix;
    VecNN *transposed;
    int w;
    int h;

public:
    MatrixNN(int, int, float*);
    MatrixNN(int, int, VecNN*);
    ~MatrixNN();
    int getWidth();
    int getHieght();
    VecNN getVec(int);
    VecNN getTransposedIndex(int);
    MatrixNN mmul(MatrixNN);
    float ApplyFilter(MatrixNN);
};


#endif //INFERENCEENGINE_MATRIXNN_H
