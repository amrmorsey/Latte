//
// Created by shadyf on 07/09/17.
//

#include "MatrixAVX.h"


//    https://stackoverflow.com/questions/13879609/horizontal-sum-of-8-packed-32bit-floats
__m256 MatrixAVX::hsums(__m256 const &v) {
    auto x = _mm256_permute2f128_ps(v, v, 1);
    auto y = _mm256_add_ps(v, x);
    x = _mm256_shuffle_ps(y, y, _MM_SHUFFLE(2, 3, 0, 1));
    x = _mm256_add_ps(x, y);
    y = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
    return _mm256_add_ps(x, y);
}
