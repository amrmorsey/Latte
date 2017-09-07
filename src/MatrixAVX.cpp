//
// Created by shadyf on 07/09/17.
//

#include "MatrixAVX.h"


inline void transpose4x4_SSE(float *A, float *B, const int lda, const int ldb) {
    __m128 row1 = _mm_load_ps(&A[0 * lda]);
    __m128 row2 = _mm_load_ps(&A[1 * lda]);
    __m128 row3 = _mm_load_ps(&A[2 * lda]);
    __m128 row4 = _mm_load_ps(&A[3 * lda]);
    _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
    _mm_store_ps(&B[0 * ldb], row1);
    _mm_store_ps(&B[1 * ldb], row2);
    _mm_store_ps(&B[2 * ldb], row3);
    _mm_store_ps(&B[3 * ldb], row4);
}

inline void transpose_block_SSE4x4(float *A, float *B, const int n, const int m, const int lda, const int ldb,
                                   const int block_size) {
#pragma omp parallel for
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < m; j += block_size) {
            int max_i2 = i + block_size < n ? i + block_size : n;
            int max_j2 = j + block_size < m ? j + block_size : m;
            for (int i2 = i; i2 < max_i2; i2 += 4) {
                for (int j2 = j; j2 < max_j2; j2 += 4) {
                    transpose4x4_SSE(&A[i2 * lda + j2], &B[j2 * ldb + i2], lda, ldb);
                }
            }
        }
    }
}