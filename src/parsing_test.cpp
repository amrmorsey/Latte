#include <chrono>
#include <ctime>
#include <ratio>
#include "Net.h"
#include "MatrixAVX.h"

using namespace std;


int main() {
    Net net("simple_what.ahsf", "weights_original", "mean");
    Matrix image = net.loadMatrix("image_original", "image");
    net.preprocess(image);
    net.setup(image);
    auto start = std::chrono::system_clock::now();
    for (size_t counter = 0; counter < 10000; ++counter)
        net.predict();

    auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start) / 10000;

    std::cout << "Completed function in " << duration.count() << " microseconds." << std::endl;

    return 0;
}