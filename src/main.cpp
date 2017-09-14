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
    net.setup(image.shape);
    net.preprocess(image);
    net.predict(image);
    std::cout <<"Predictions\n" << net.layers[net.layers.size()-1]->output << std::endl;
    return 0;
}