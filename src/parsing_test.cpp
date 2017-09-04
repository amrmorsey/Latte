#include <chrono>
#include "Net.h"

using namespace std;


int main() {
    Net net("simple_what.ahsf", "weights", "mean");
    Matrix image = net.loadMatrix("image_original", "image");
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    net.predict(image);
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Time difference(microseconds) = " << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() <<std::endl;
    return 0;
}
