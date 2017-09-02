//
// Created by Amr on 8/24/17.
//

#include "VecNN.h"

VecNN::VecNN(int s, const float * arr) {
    size = s;
    N = size - (size%4);
    if(size%4 != 0)
        N = (N + 4);
    N = N/4;
    vec = new __m128[N];
    fvec = new float[size];

    for(int i = 0; i< N; i++){
        vec[i] = _mm_load_ps(&arr[4*i]);
    }
}


VecNN::VecNN(int s) {
    size = s;
    N = size - (size%4);
    if(size%4 != 0)
        N = (N + 4);
    N = N/4;
    vec = new __m128[N];
    fvec = new float[size];
}

VecNN::~VecNN() {

}

__m128 VecNN::getAtIndex(int i) {
    if(i < N)
        return vec[i];
    else{
        const float e = -1;
        __m128 error = _mm_load1_ps(&e);
        return error;
    }
}

int VecNN::getSize() {
    return N;
}

float VecNN::dot(VecNN& a) {
    if(this->getSize() == a.getSize()){
        __m128 *res = new __m128[this->getSize()];
        for (int i = 0; i < this->getSize(); i++) {
            res[i] = _mm_dp_ps(this->getAtIndex(i), a.getAtIndex(i), 0xff);
        }
        float result = 0;
        for (int j = 0; j < this->getSize(); ++j) {
            float x[4];
            _mm_store_ps(x, res[j]);
            result +=x[0];
        }
        return result;
    }
}

int VecNN::getNoOfElements() {
    return size;
}

void VecNN::loadData(const float *arr) {
    for(int i = 0; i< N; i++){
        vec[i] = _mm_load_ps(&arr[4*i]);
    }
}

void VecNN::setAtIndex(int index, __m128 a) {
    if(index < N){
        vec[index] = a;
    }
}

float * VecNN::sub(VecNN& a) {
    if(this->getSize() == a.getSize()){
        __m128 *res = new __m128[this->getSize()];
        for (int i = 0; i < this->getSize(); i++) {
            res[i] = _mm_sub_ps(this->getAtIndex(i), a.getAtIndex(i));
        }
        float *ress = new float[this->size*4];

        for (int j = 0; j <this->getSize() ; j++) {
            float x[4];
            _mm_store_ps(x, res[j]);
            ress[4*j + 0] = x[0];
            ress[4*j + 1] = x[1];
            ress[4*j + 2] = x[2];
            ress[4*j + 3] = x[3];
        }
        return ress;
    }
}

float *VecNN::add(VecNN& a) {
    if(this->getSize() == a.getSize()){
        __m128 *res = new __m128[this->getSize()];
        for (int i = 0; i < this->getSize(); i++) {
            res[i] = _mm_add_ps(this->getAtIndex(i), a.getAtIndex(i));
        }
        float *ress = new float[this->size*4];

        for (int j = 0; j <this->getSize() ; j++) {
            float x[4];
            _mm_store_ps(x, res[j]);
            ress[4*j + 0] = x[0];
            ress[4*j + 1] = x[1];
            ress[4*j + 2] = x[2];
            ress[4*j + 3] = x[3];
        }
        return ress;
    }
}

