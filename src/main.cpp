#include <chrono>
#include <ctime>
#include <ratio>
#include "Net.h"
#include "MatrixAVX.h"
#include <iostream>
#include "utils.h"

using namespace std;


int main() {
    Net net("simple_test28Gray.ahsf", "weights", "mean");
    MatrixAVX image = loadMatrix("image", "image");
    net.preprocess(image);
    net.setup(image);
    auto start = std::chrono::system_clock::now();
    for (size_t counter = 0; counter < 100000; ++counter)
        net.predict(image);
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start) / 100000;
    std::cout << "Completed function in " << duration.count() << " microseconds." << std::endl;
    std::cout <<"Predictions\n" << net.layers[net.layers.size()-1]->output << std::endl;
    return 0;
}