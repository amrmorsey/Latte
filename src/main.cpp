//#include <iostream>
//#include <vector>
//#include <emmintrin.h>
//#include <intrin.h>
//using namespace std;
//
//void feedForward(int (*inp)[7][7], int (*ws)[3][3][3], int* bs, int stride, int filter){
//    int out[2][3][3];
//    int count = 0;
//    int count2 = 0;
//    for(int i = 1; i < 7-int(filter/2); i= i+stride){
//        for(int j=1; j<7-int(filter/2); j = j+stride){
//            out[0][count2][count] = inp[0][i-1][j-1] * ws[0][0][0][0] +
//                                    inp[0][i-1][j] * ws[0][0][0][1] +
//                                    inp[0][i-1][j+1] * ws[0][0][0][2] +
//                                    inp[0][i][j-1] * ws[0][0][1][0] +
//                                    inp[0][i][j] * ws[0][0][1][1] +
//                                    inp[0][i][j+1] * ws[0][0][1][2] +
//                                    inp[0][i+1][j-1] * ws[0][0][2][0] +
//                                    inp[0][i+1][j] * ws[0][0][2][1] +
//                                    inp[0][i+1][j+1] * ws[0][0][2][2]
//                                    + inp[1][i-1][j-1] * ws[0][1][0][0] +
//                                    inp[1][i-1][j] * ws[0][1][0][1] +
//                                    inp[1][i-1][j+1] * ws[0][1][0][2] +
//                                    inp[1][i][j-1] * ws[0][1][1][0] +
//                                    inp[1][i][j] * ws[0][1][1][1] +
//                                    inp[1][i][j+1] * ws[0][1][1][2] +
//                                    inp[1][i+1][j-1] * ws[0][1][2][0] +
//                                    inp[1][i+1][j] * ws[0][1][2][1] +
//                                    inp[1][i+1][j+1] * ws[0][1][2][2]
//                                    +inp[2][i-1][j-1] * ws[0][2][0][0] +
//                                    inp[2][i-1][j] * ws[0][2][0][1] +
//                                    inp[2][i-1][j+1] * ws[0][2][0][2] +
//                                    inp[2][i][j-1] * ws[0][2][1][0] +
//                                    inp[2][i][j] * ws[0][2][1][1] +
//                                    inp[2][i][j+1] * ws[0][2][1][2] +
//                                    inp[2][i+1][j-1] * ws[0][2][2][0] +
//                                    inp[2][i+1][j] * ws[0][2][2][1] +
//                                    inp[2][i+1][j+1] * ws[0][2][2][2]
//                                    + bs[0];
//
//            out[1][count2][count] = inp[0][i-1][j-1] * ws[1][0][0][0] +
//                                    inp[0][i-1][j] * ws[1][0][0][1] +
//                                    inp[0][i-1][j+1] * ws[1][0][0][2] +
//                                    inp[0][i][j-1] * ws[1][0][1][0] +
//                                    inp[0][i][j] * ws[1][0][1][1] +
//                                    inp[0][i][j+1] * ws[1][0][1][2] +
//                                    inp[0][i+1][j-1] * ws[1][0][2][0] +
//                                    inp[0][i+1][j] * ws[1][0][2][1] +
//                                    inp[0][i+1][j+1] * ws[1][0][2][2]
//                                    + inp[1][i-1][j-1] * ws[1][1][0][0] +
//                                    inp[1][i-1][j] * ws[1][1][0][1] +
//                                    inp[1][i-1][j+1] * ws[1][1][0][2] +
//                                    inp[1][i][j-1] * ws[1][1][1][0] +
//                                    inp[1][i][j] * ws[1][1][1][1] +
//                                    inp[1][i][j+1] * ws[1][1][1][2] +
//                                    inp[1][i+1][j-1] * ws[1][1][2][0] +
//                                    inp[1][i+1][j] * ws[1][1][2][1] +
//                                    inp[1][i+1][j+1] * ws[1][1][2][2]
//                                    +inp[2][i-1][j-1] * ws[1][2][0][0] +
//                                    inp[2][i-1][j] * ws[1][2][0][1] +
//                                    inp[2][i-1][j+1] * ws[1][2][0][2] +
//                                    inp[2][i][j-1] * ws[1][2][1][0] +
//                                    inp[2][i][j] * ws[1][2][1][1] +
//                                    inp[2][i][j+1] * ws[1][2][1][2] +
//                                    inp[2][i+1][j-1] * ws[1][2][2][0] +
//                                    inp[2][i+1][j] * ws[1][2][2][1] +
//                                    inp[2][i+1][j+1] * ws[1][2][2][2]
//                                    + bs[1];
//            count++;
//        }
//        count2++;
//        count = 0;
//    }
//    for (int i = 0; i < 2; ++i) {
//        for (int j = 0; j < 3; ++j) {
//            for (int k = 0; k < 3; ++k) {
//                cout<<out[i][j][k]<<" ";
//            }
//            cout<<endl;
//        }
//        cout<<endl;
//    }
//}
//
//void testing(vector<int> in, vector<int> w, vector<int> b, int stride, int filter, int in_size, int pad, int no_of_filters){
//    int out_depth = no_of_filters;
//    int out_hw = ((in_size-filter + 2*pad)/stride) + 1;
//
//    for (int i = pad*in_size+1; i < in.size(); i = i + stride*in_size) {
//        for (int j = i; j < in_size; j = j + stride) {
//
//        }
//
//    }
//}
//
//int main() {
//    std::cout << "Hello, World!" << std::endl;
//    int F = 3;
//    int S = 2;
//    int inp [3][7][7] = {{{0,0,0,0,0,0,0}, {0,0,1,1,0,2,0}, {0,0,1,2,1,1,0}, {0,2,1,1,2,1,0}, {0,1,1,1,2,1,0} , {0,0,1,2,0,2,0}, {0,0,0,0,0,0,0}},
//                         {{0,0,0,0,0,0,0}, {0,2,0,2,0,2,0}, {0,1,0,1,1,0,0}, {0,1,0,2,0,1,0}, {0,0,1,1,0,0,0} , {0,2,2,2,0,0,0}, {0,0,0,0,0,0,0}},
//                         {{0,0,0,0,0,0,0}, {0,0,1,1,1,1,0}, {0,2,0,0,2,2,0}, {0,1,1,1,1,1,0}, {0,0,0,1,0,2,0} , {0,1,1,2,2,0,0}, {0,0,0,0,0,0,0}}};
//
//    vector<vector<vector<int>>> input = {{{0,0,0,0,0,0,0}, {0,0,1,1,0,2,0}, {0,0,1,2,1,1,0}, {0,2,1,1,2,1,0}, {0,1,1,1,2,1,0} , {0,0,1,2,0,2,0}, {0,0,0,0,0,0,0}},
//                                         {{0,0,0,0,0,0,0}, {0,2,0,2,0,2,0}, {0,1,0,1,1,0,0}, {0,1,0,2,0,1,0}, {0,0,1,1,0,0,0} , {0,2,2,2,0,0,0}, {0,0,0,0,0,0,0}},
//                                         {{0,0,0,0,0,0,0}, {0,0,1,1,1,1,0}, {0,2,0,0,2,2,0}, {0,1,1,1,1,1,0}, {0,0,0,1,0,2,0} , {0,1,1,2,2,0,0}, {0,0,0,0,0,0,0}}};
//
//    int w [2][3][3][3] = {{{{1,1,0},{1, -1, -1}, {1, 1 ,1}}, {{1,-1,0},{-1, 1, -1}, {0, -1 ,0}}, {{-1,0,0},{-1, 0, 0}, {-1, -1 ,-1}}},
//                          {{{0,0,1},{-1, 0, 1}, {-1, -1 ,-1}}, {{1,0,1},{-1, 1, -1}, {-1, 1 ,-1}}, {{0,-1,1},{1, 1, 0}, {0, -1 ,1}}}};
//
//    vector<vector<vector<vector<int>>>> weights = {{{{1,1,0},{1, -1, -1}, {1, 1 ,1}}, {{1,-1,0},{-1, 1, -1}, {0, -1 ,0}}, {{-1,0,0},{-1, 0, 0}, {-1, -1 ,-1}}},
//                                                   {{{0,0,1},{-1, 0, 1}, {-1, -1 ,-1}}, {{1,0,1},{-1, 1, -1}, {-1, 1 ,-1}}, {{0,-1,1},{1, 1, 0}, {0, -1 ,1}}}};
////    feedVector(&weights);
//    int bias[2] = {1,0};
//    vector<int> b2 = {1,0};
//
//    vector<int> inp2 = {0,0,0,0,0,0,0,0,0,1,1,0,2,0,0,0,1,2,1,1,0,0,2,1,1,2,1,0,0,1,1,1,2,1,0,0,0,1,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,2,0,2,0,0,1,0,1,1,0,0,0,1,0,2,0,1,0,0,0,1,1,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,0,
//                  0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,2,0,0,2,2,0,0,1,1,1,1,1,0,0,0,0,1,0,2,0,0,1,1,2,2,0,0,0,0,0,0,0,0,0};
//    vector<int> w2 = {1,1,0,1, -1, -1,1, 1 ,1,1,-1,0,-1, 1, -1,0, -1 ,0,-1,0,0,-1, 0, 0,-1, -1 ,-1,0,0,1,-1, 0, 1,-1, -1 ,-1,1,0,1,-1, 1, -1,-1, 1 ,-1,0,-1,1,1, 1, 0,0, -1 ,1};
//
//    int i = 1;
//    int j = 5;
//    feedForward(inp, w, bias, S,F);
//    return 0;
//}

#include <immintrin.h>
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <string.h>
#include "VecNN.h"
#include "Matrix.h"
#include "VecAVX.h"

union Mat44 {
    float m[4][4];
    __m128 row[4];
};

// reference implementation
void matmult_ref(Mat44 &out, const Mat44 &A, const Mat44 &B)
{
    Mat44 t; // write to temp
    for (int i=0; i < 4; i++)
        for (int j=0; j < 4; j++)
            t.m[i][j] = A.m[i][0]*B.m[0][j] + A.m[i][1]*B.m[1][j] + A.m[i][2]*B.m[2][j] + A.m[i][3]*B.m[3][j];

    out = t;
}

// linear combination:
// a[0] * B.row[0] + a[1] * B.row[1] + a[2] * B.row[2] + a[3] * B.row[3]
static inline __m128 lincomb_SSE(const __m128 &a, const Mat44 &B)
{
    __m128 result;
    result = _mm_mul_ps(_mm_shuffle_ps(a, a, 0x00), B.row[0]);
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0x55), B.row[1]));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0xaa), B.row[2]));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0xff), B.row[3]));
//    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0x154), B.row[4]));
    return result;
}

// this is the right approach for SSE ... SSE4.2
void matmult_SSE(Mat44 &out, const Mat44 &A, const Mat44 &B)
{
    // out_ij = sum_k a_ik b_kj
    // => out_0j = a_00 * b_0j + a_01 * b_1j + a_02 * b_2j + a_03 * b_3j
    __m128 out0x = lincomb_SSE(A.row[0], B);
    __m128 out1x = lincomb_SSE(A.row[1], B);
    __m128 out2x = lincomb_SSE(A.row[2], B);
    __m128 out3x = lincomb_SSE(A.row[3], B);
    //__m128 out4x = lincomb_SSE(A.row[4], B);

    out.row[0] = out0x;
    out.row[1] = out1x;
    out.row[2] = out2x;
    out.row[3] = out3x;
   // out.row[4] = out4x;
}

// another linear combination, using AVX instructions on XMM regs
static inline __m128 lincomb_AVX_4mem(const float *a, const Mat44 &B)
{
    __m128 result;
    result = _mm_mul_ps(_mm_broadcast_ss(&a[0]), B.row[0]);
    result = _mm_add_ps(result, _mm_mul_ps(_mm_broadcast_ss(&a[1]), B.row[1]));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_broadcast_ss(&a[2]), B.row[2]));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_broadcast_ss(&a[3]), B.row[3]));
    return result;
}

// using AVX instructions, 4-wide
//// this can be better if A is in memory.
void matmult_AVX_4mem(Mat44 &out, const Mat44 &A, const Mat44 &B)
{
    _mm256_zeroupper();
    __m128 out0x = lincomb_AVX_4mem(A.m[0], B);
    __m128 out1x = lincomb_AVX_4mem(A.m[1], B);
    __m128 out2x = lincomb_AVX_4mem(A.m[2], B);
    __m128 out3x = lincomb_AVX_4mem(A.m[3], B);

    out.row[0] = out0x;
    out.row[1] = out1x;
    out.row[2] = out2x;
    out.row[3] = out3x;
}

// dual linear combination using AVX instructions on YMM regs
static inline __m256 twolincomb_AVX_8(__m256 A01, const Mat44 &B)
{
    __m256 result;
    result = _mm256_mul_ps(_mm256_shuffle_ps(A01, A01, 0x00), _mm256_broadcast_ps(&B.row[0]));
    result = _mm256_add_ps(result, _mm256_mul_ps(_mm256_shuffle_ps(A01, A01, 0x55), _mm256_broadcast_ps(&B.row[1])));
    result = _mm256_add_ps(result, _mm256_mul_ps(_mm256_shuffle_ps(A01, A01, 0xaa), _mm256_broadcast_ps(&B.row[2])));
    result = _mm256_add_ps(result, _mm256_mul_ps(_mm256_shuffle_ps(A01, A01, 0xff), _mm256_broadcast_ps(&B.row[3])));
    return result;
}

// this should be noticeably faster with actual 256-bit wide vector units (Intel);
// not sure about double-pumped 128-bit (AMD), would need to check.
void matmult_AVX_8(Mat44 &out, const Mat44 &A, const Mat44 &B)
{
    _mm256_zeroupper();
    __m256 A01 = _mm256_loadu_ps(&A.m[0][0]);
    __m256 A23 = _mm256_loadu_ps(&A.m[2][0]);

    __m256 out01x = twolincomb_AVX_8(A01, B);
    __m256 out23x = twolincomb_AVX_8(A23, B);

    _mm256_storeu_ps(&out.m[0][0], out01x);
    _mm256_storeu_ps(&out.m[2][0], out23x);
}

// ---- testing stuff

static float randf()
{
    // assumes VC++ rand()
    return (rand() - 16384.0f) / 1024.0f;
}

static void randmat(Mat44 &M)
{
    for (int i=0; i < 4; i++)
        for (int j=0; j < 4; j++)
            M.m[i][j] = randf();
}

int the_mask = 0; // global so the compiler can't be sure what its value is for opt.

static void run_ref(Mat44 *out, const Mat44 *A, const Mat44 *B, int count)
{
    for (int i=0; i < count; i++)
    {
        int j = i & the_mask;
        matmult_ref(out[j], A[j], B[j]);
    }
}

static void run_SSE(Mat44 *out, const Mat44 *A, const Mat44 *B, int count)
{
    for (int i=0; i < count; i++)
    {
        int j = i & the_mask;
        matmult_SSE(out[j], A[j], B[j]);
    }
}

static void run_AVX_4mem(Mat44 *out, const Mat44 *A, const Mat44 *B, int count)
{
    for (int i=0; i < count; i++)
    {
        int j = i & the_mask;
        matmult_AVX_4mem(out[j], A[j], B[j]);
    }
}

static void run_AVX_8(Mat44 *out, const Mat44 *A, const Mat44 *B, int count)
{
    for (int i=0; i < count; i++)
    {
        int j = i & the_mask;
        matmult_AVX_8(out[j], A[j], B[j]);
    }
}


struct vec4
{
    __m128 xmm;

    vec4 (__m128 v) : xmm (v) {}

    vec4 (float v) { xmm = _mm_set1_ps(v); }

    vec4 (float x, float y, float z, float w)
    { xmm = _mm_set_ps(w,z,y,x); }

    vec4 (const float *v) { xmm = _mm_load_ps(v); }

    vec4 operator* (const vec4 &v) const
    { return vec4(_mm_mul_ps(xmm, v.xmm)); }

    vec4 operator+ (const vec4 &v) const
    { return vec4(_mm_add_ps(xmm, v.xmm)); }

    vec4 operator- (const vec4 &v) const
    { return vec4(_mm_sub_ps(xmm, v.xmm)); }

    vec4 operator/ (const vec4 &v) const
    { return vec4(_mm_div_ps(xmm, v.xmm)); }

    void operator*= (const vec4 &v)
    { xmm = _mm_mul_ps(xmm, v.xmm); }

    void operator+= (const vec4 &v)
    { xmm = _mm_add_ps(xmm, v.xmm); }

    void operator-= (const vec4 &v)
    { xmm = _mm_sub_ps(xmm, v.xmm); }

    void operator/= (const vec4 &v)
    { xmm = _mm_div_ps(xmm, v.xmm); }

    void operator>> (float *v)
    { _mm_store_ps(v, xmm); }

};

void mmul_vec4(const float * a, const float * b, float * r)
{
    for (int i=0; i<16; i+=4) {
        vec4 rl = vec4(a) * vec4(b[i]);
        int n4 = 25 - (25 % 4);
        for (int j=1; j<4; j++)
            rl += vec4(&a[j*4]) * vec4(b[i+j]);
        rl >> &r[i];
    }
}

void mmul_nxn(const float *a, const float *b, float *r, int n, int row){
    float zero[row];
    zero[0] = 0;
    zero[1] = 0;
    zero[2] = 0;
    zero[3] = 0;
    for (int i = 0; i <n ; i=i+row) {
        float temp[row];
        int c = 0;
        __m128 acc = _mm_load_ps(zero);
        float sum = 0;
        int n4 = n - (n%row);
        int k;
        //acc = _mm_mul_ps(_mm_load_ps(&a[i]),_mm_load_ps(&b[i+k*4]));
        for(k = 0; k<4;k++){
            __m128 X = _mm_load_ps(&a[i]);
            __m128 Y = _mm_load_ps(&b[k*4]);
            acc = _mm_add_ps(acc, _mm_mul_ps(X,Y));
        }
        _mm_store_ps(temp, acc);
//        sum = temp[0] + temp[1] + temp[2] + temp[3];
        //non vectorise here
        for (k = n4; k < n; k++) {
            sum = sum + a[i] * b[k];
        }

        for(int j = 0; j<row; j++){
            r[i+j] = temp[j];
        }

    }
}


int main(int argc, char **argv)
{
//    static const struct {
//        const char *name;
//        void (*matmult)(Mat44 &out, const Mat44 &A, const Mat44 &B);
//    } variants[] = {
//            { "ref",      matmult_ref },
//            { "SSE",      matmult_SSE },
//            { "AVX_4mem", matmult_AVX_4mem },
//            { "AVX_8",    matmult_AVX_8 },
//    };
//    static const int nvars = (int) (sizeof(variants) / sizeof(*variants));
//
//    srand(1234); // deterministic random tests(TM)
//
//    // correctness tests
//    // when compiled with /arch:SSE (or SSE2/AVX), all functions are
//    // supposed to return the exact same results!
//    for (int i=0; i < 1000000; i++)
//    {
//        Mat44 A, B, out, ref_out;
//        randmat(A);
//        randmat(B);
//        matmult_ref(ref_out, A, B);
//
//        for (int j=0; j < nvars; j++)
//        {
//            variants[j].matmult(out, A, B);
//            if (memcmp(&out, &ref_out, sizeof(out)) != 0)
//            {
//                fprintf(stderr, "%s fails test\n", variants[j].name);
//                exit(1);
//            }
//        }
//    }
//
//    printf("all ok.\n");
//
//    // perf tests
//    // as usual with such microbenchmarks, this isn't measuring anything
//    // terribly useful, but here goes.
//    static const struct {
//        const char *name;
//        void (*run)(Mat44 *out, const Mat44 *A, const Mat44 *B, int count);
//    } perf_variants[] = {
//            { "ref",      run_ref },
//            { "SSE",      run_SSE },
//            { "AVX_4mem", run_AVX_4mem },
//            { "AVX_8",    run_AVX_8 },
//    };
//    static const int nperfvars = (int) (sizeof(perf_variants) / sizeof(*perf_variants));
//
//    /*
//       results on my sandy bridge laptop when compiling the code in x64
//       mode with VC2010 using /arch:AVX:
//        all ok.
//                 ref: 59.00 cycles
//                 SSE: 20.52 cycles
//            AVX_4mem: 15.64 cycles
//               AVX_8: 14.13 cycles
//    */
//
//    Mat44 Aperf, Bperf, out;
//    randmat(Aperf);
//    randmat(Bperf);
//
//    for (int i=0; i < nvars; i++)
//    {
//        static const int nruns = 4096;
//        static const int muls_per_run = 4096;
//        unsigned long long best_time = ~0ull;
//
//        for (int run=0; run < nruns; run++)
//        {
//            unsigned long long time = __rdtsc();
//            perf_variants[i].run(&out, &Aperf, &Bperf, muls_per_run);
//            time = __rdtsc() - time;
//            if (time < best_time)
//                best_time = time;
//        }
//
//        double cycles_per_run = (double) best_time / (double) muls_per_run;
//        printf("%12s: %.2f cycles\n", perf_variants[i].name, cycles_per_run);
//    }

//    Mat44 A, B, out;
//    A.m[0][0] = 1;
//    A.m[0][1] = 2;
//    A.m[0][2] = 3;
//    A.m[0][3] = 4;
//    A.m[0][4] = 5;
//    A.m[1][0] = 6;
//    A.m[1][1] = 7;
//    A.m[1][2] = 8;
//    A.m[1][3] = 9;
//    A.m[2][4] = 10;
//    A.m[2][0] = 11;
//    A.m[2][1] = 12;
//    A.m[2][2] = 13;
//    A.m[2][3] = 14;
//    A.m[2][4] = 15;
//    A.m[3][0] = 16;
//    A.m[3][1] = 17;
//    A.m[3][2] = 18;
//    A.m[3][3] = 19;
//    A.m[3][4] = 20;
////    A.m[4][0] = 21;
////    A.m[4][1] = 22;
////    A.m[4][2] = 23;
////    A.m[4][3] = 24;
////    A.m[4][4] = 25;
//
//    B.m[0][0] = 1;
//    B.m[0][1] = 0;
//    B.m[0][2] = 0;
//    B.m[0][3] = 0;
//    B.m[0][4] = 0;
//    B.m[1][0] = 0;
//    B.m[1][1] = 1;
//    B.m[1][2] = 0;
//    B.m[1][3] = 0;
//    B.m[2][4] = 0;
//    B.m[2][0] = 0;
//    B.m[2][1] = 0;
//    B.m[2][2] = 1;
//    B.m[2][3] = 0;
//    B.m[2][4] = 0;
//    B.m[3][0] = 0;
//    B.m[3][1] = 0;
//    B.m[3][2] = 0;
//    B.m[3][3] = 1;
//    B.m[3][4] = 0;
////    B.m[4][0] = 0;
////    B.m[4][1] = 0;
////    B.m[4][2] = 0;
////    B.m[4][3] = 0;
////    B.m[4][4] = 1;

    float c[] = {1,2,3,4,5,6,7,8,8,7,6,5,4,3,2,1};
    float d[] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};

    float a[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25};
    float b[] = {25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
    float r[16];
//    mmul_vec4(a, b,r);
//    matmult_SSE(out, A, B);
  //  mmul_nxn(c, d, r,16, 4);

//    for (int i = 0; i < 16; i= i+4) {
//        for (int j = 0; j < 4; j++) {
//            printf("%f\t", r[i+j]);
//        }
//        printf("\n");
//
//    }

    //float aa[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    //float bb[4] = {0.5, 1.5, 2.5, 3.5};
    vector<float>  aa(51);
    vector<float> bb(51);
    for(int i = 0; i< 51; i++){
        aa[i] = i;
    }
    for(int i = 50; i>=0; i--){
        bb[50-i] = i;
    }
//    VecAVX vv(51, aa);
//    VecAVX vf(51, bb);
    vector<int> size = {51};
    Matrix vv(size);
    Matrix vf(size);
    vv.matrix = aa;
    vf.matrix = bb;
    Matrix oo = vv.sub(vf);
    cout<<"Here"<<endl;

//    unsigned long long time = __rdtsc();
//    VecAVX vv(51, aa);
//    VecAVX vf(51, bb);
//    vv.dot(vf);
//    time = __rdtsc() - time;
//    cout<<"AVX dot product Time in seconds: "<<time<<endl;
//
//    time = __rdtsc();
//    VecNN vn(51, aa);
//    VecNN fn(51, bb);
//    vn.dot(fn);
//    time = __rdtsc() - time;
//    cout<<"SSE dot product Time in seconds: "<<time<<endl;

//    __m128 xx = _mm_load_ps(aa);
//    __m128  yy = _mm_load_ps(bb);
//    __m128 *arr;
//    arr = new __m128[2];
    //const double __xx = aa[4];
    //__m128 zz = _mm_load_ps(&aa[4]);
    //_mm_insert_ps()
    //__m128 oo = _mm_dp_ps(xx, yy, 0xff);
    //VecNN vv(4, aa);
    //VecNN vf(4, bb);
    //printf("%f", vv.dot(vf));
//    vector<float> matrix = {2,0,2,2,2,
//                            1,0,1,0,2,
//                            2,0,1,0,0,
//                            0,2,1,2,0,
//                            0,2,2,1,0,
//
//                            0,2,1,2,0,
//                            2,1,1,1,2,
//                            2,2,1,0,1,
//                            1,2,1,1,1,
//                            0,0,1,0,1,
//
//                            1,1,1,0,0,
//                            1,0,2,1,0,
//                            2,2,1,1,2,
//                            1,2,0,0,1,
//                            0,0,1,1,0};
//    vector<int> shape= {5,5,3};
//    Matrix mm(matrix, shape);
//    vector<int> filter = {3,3,3,2};
////    Matrix x = mm.im2col(filter, 2);
//    vector<float> mFilter = {-1,1,-1,0,0,-1,-1,0,1,0,0,1,1,1,1,0,1,-1,-1,1,0,0,0,0,1,1,1,1,-1,-1,1,1,0,0,-1,0,-1,-1,-1,1,1,1,-1,1,1,1,1,0,-1,-1,1,0,1,-1};
//    Matrix w(mFilter, filter);
//    Matrix xx = mm.MaxRow(2, 2, 0);
//    x.dot(w);
//    __m256 x = _mm256_load_ps(aa);
    return 0;
}