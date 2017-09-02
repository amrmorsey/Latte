//
// Created by Amr on 8/24/17.
//

#ifndef INFERENCEENGINE_VECNN_H
#define INFERENCEENGINE_VECNN_H
#include <immintrin.h>
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <string.h>

class VecNN {
private:
    int N;
    __m128 *vec;
    float *fvec;
    int size;
public:
    VecNN(int, const float*);
    VecNN(int);
    ~VecNN();
    float dot(VecNN&);
    __m128 getAtIndex(int);
    int getSize();
    int getNoOfElements();
    void loadData(const float*);
    void setAtIndex(int, __m128);
    float* sub(VecNN&);
    float* add(VecNN&);
};


#endif //INFERENCEENGINE_VECNN_H
