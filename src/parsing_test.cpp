#include <chrono>
#include <ctime>
#include <ratio>
#include "Net.h"
#include "MatrixAVX.h"

using namespace std;


int main() {
    Net net("simple_what.ahsf", "weights_original", "mean");
    Matrix image = net.loadMatrix("image_original", "image");
    MatrixAVX what({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.2}, {2, 5});
    __m256 odds = _mm256_set_ps(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 6.0);

//    __m256 c =  _mm256_mul_ps(evens, odds);
//    cout <<  _mm256_cvtss_f32(hsums(c));
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