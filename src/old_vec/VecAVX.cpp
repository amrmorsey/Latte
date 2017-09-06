//
// Created by Amr on 8/31/17.
//

#include "VecAVX.h"

VecAVX::VecAVX(int s, float * arr) {
    size = s;
    N = size - (size % 8);
    if (size % 8 != 0)
        N = (N + 8);
    paddedSize = N;
    N = N / 8;
    vec = new __m256[N];
    fvec = new float[size];
    for (int j = size; j < paddedSize; j++) {
        arr[j] = 0;
    }
    for (int i = 0; i < N; i++) {
        vec[i] = _mm256_load_ps(&arr[8*i]);
    }
}

VecAVX::VecAVX(int s) {
    size = s;
    N = size - (size%8);
    if(size%8 != 0)
        N = (N + 8);
    N = N/8;
    vec = new __m256[N];
    fvec = new float[size];
}

VecAVX::~VecAVX() {

}

float VecAVX::dot(VecAVX a) {
    if(this->getSize() == a.getSize()){
        //__m256 *res = (__m256 *)_mm_malloc(this->getSize(),256);
        __m256 *res = new __m256[this->getSize()];
        for (int i = 0; i < this->getSize(); i++) {
            res[i] = _mm256_dp_ps(this->vec[i], a.vec[i], 0xff);
        }
        float result = 0;
        for (int j = 0; j < this->getSize(); ++j) {
            float x[8];
            _mm256_store_ps(x, res[j]);
            result +=x[0] + x[4];
        }
        return result;
    }
}
//not working for some reason
//__m256 VecAVX::getAtIndex(int i) {
//    if(i < N)
//        return vec[i], vec[i+1];
//    else{
//        const float e = -1;
//        __m256 error = _mm256_broadcast_ss(&e);
//        return error;
//    }
//}

int VecAVX::getSize() {
    return N;
}

int VecAVX::getNoOfElements() {
    return size;
}

void VecAVX::loadData(const float * arr) {
    for(int i = 0; i< N; i++){
        vec[i] = _mm256_load_ps(&arr[8*i]);
    }
}

void VecAVX::setAtIndex(int index, __m256 a) {
    if (index < N) {
        vec[index] = a;
    }
}







