//
// Created by Amr on 8/31/17.
//

#ifndef INFERENCEENGINE_VECAVX_H
#define INFERENCEENGINE_VECAVX_H

#include <immintrin.h>
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <string.h>

class VecAVX {
private:
    int N;
    //__m256 *vec;
    float *fvec;
    int size;
    int paddedSize;
public:
    __m256 *vec;
    VecAVX(int, float*);
    VecAVX(int);
    ~VecAVX();
    float dot(VecAVX);
    //__m256 getAtIndex(int);
    int getSize();
    int getNoOfElements();
    void loadData(const float*);
    void setAtIndex(int, __m256);
};


#endif //INFERENCEENGINE_VECAVX_H
