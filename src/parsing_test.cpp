#include <chrono>
#include <ctime>
#include <ratio>
#include "Net.h"
#include "MatrixAVX.h"
#include <iostream>
#include "utils.h"

using namespace std;


int main() {
    Net net("simple_what.ahsf", "weights_original", "mean");
    MatrixAVX image = loadMatrix("image_original", "image");
    MatrixAVX x({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, {2, 5});
    MatrixAVX y({1.0, 2.0, 3.0, 4.0, 5.0}, {5, 1});
    MatrixAVX out({5, 1});
    x.dot_product(y, out);
    std::cout << out;

//    __m256 odds = _mm256_set_ps(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 6.0);

//    __m256 c =  _mm256_mul_ps(evens, odds);
//    cout <<  _mm256_cvtss_f32(hsums(c));
//    net.preprocess(image);
//    net.predict(image);
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