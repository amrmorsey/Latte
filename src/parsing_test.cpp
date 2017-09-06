#include <chrono>
#include <ctime>
#include <ratio>
#include "Net.h"

using namespace std;

union Mat44 {
    float *m;
    __m128 *row;
};

int main() {
    Net net("simple_what.ahsf", "weights_original", "mean");
    Matrix image = net.loadMatrix("image_original", "image");
    auto a = Mat44();
    a.m = new float[8];

    for(int i = 0 ; i < 8; i++){
        a.m[i] = float(2.0);
    }
    __m128 c =  _mm_mul_ps(a.row[0], a.row[1]);
    __m128 t1 = _mm_hadd_ps(c,c);
    __m128 t2 = _mm_hadd_ps(t1,t1);
    cout <<  _mm_cvtss_f32(t2);
//    net.preprocess(image);
//    auto start = std::chrono::system_clock::now();
//    for (size_t counter = 0; counter < 10000; ++counter)
//        net.predict(image);
//
//    auto duration =
//            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start) / 10000;
//
//    std::cout << "Completed function in " << duration.count() << " microseconds." << std::endl;

    return 0;
}