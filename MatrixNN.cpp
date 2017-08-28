//
// Created by Amr on 8/23/17.
//

#include "MatrixNN.h"

MatrixNN::MatrixNN(int height, int width, float *inMat) {
    h = height;
    w = width;

//    matrix = new VecNN[h](w);

    for (int i = 0; i < h; i++) {
        matrix[i].loadData(&inMat[w*i]);
    }
}

MatrixNN::MatrixNN(int height, int width, VecNN *v) {
    h = height;
    w = width;
    matrix = v;
}

MatrixNN::~MatrixNN() {

}

int MatrixNN::getHieght() {
    return h;
}

int MatrixNN::getWidth() {
    return w;
}

VecNN MatrixNN::getVec(int index) {
    if(index < h){
        return matrix[index];
    }
}

float MatrixNN::ApplyFilter(MatrixNN m) {
    if(this->getHieght() == m.getHieght() && this->getWidth() == m.getWidth()){
        float result = 0;
        for (int i = 0; i < this->getHieght(); i++) {
            result += this->getVec(i).dot(m.getVec(i));
        }
        return result;
    }
}

// linear combination:
// a[0] * B.row[0] + a[1] * B.row[1] + a[2] * B.row[2] + a[3] * B.row[3]
static inline __m128 lincomb_SSE(const __m128 &a, VecNN &B)
{
    __m128 result;
    result = _mm_mul_ps(_mm_shuffle_ps(a, a, 0x00), B.getAtIndex(0));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0x55), B.getAtIndex(1)));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0xaa), B.getAtIndex(2)));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0xff), B.getAtIndex(3)));
    return result;
}

// this is the right approach for SSE ... SSE4.2
void matmult_SSE(VecNN &out,  VecNN &A,  VecNN &B)
{
    // out_ij = sum_k a_ik b_kj
    // => out_0j = a_00 * b_0j + a_01 * b_1j + a_02 * b_2j + a_03 * b_3j
    __m128 out0x = lincomb_SSE(A.getAtIndex(0), B);
    __m128 out1x = lincomb_SSE(A.getAtIndex(1), B);
    __m128 out2x = lincomb_SSE(A.getAtIndex(2), B);
    __m128 out3x = lincomb_SSE(A.getAtIndex(3), B);
    //__m128 out4x = lincomb_SSE(A.row[4], B);

    out.setAtIndex(0,out0x);
    out.setAtIndex(1,out1x);
    out.setAtIndex(2,out2x);
    out.setAtIndex(3,out3x);
    // out.row[4] = out4x;
}
