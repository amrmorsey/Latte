#include <chrono>
#include <ctime>
#include <ratio>
#include "Net.h"
#include "MatrixAVX.h"
#include <iostream>
#include "utils.h"
#include <iterator>

using namespace std;


int main() {
    Net net("simple_test28Gray.ahsf", "weights", "mean");
    MatrixAVX image1 = loadImage("1.png");
    MatrixAVX image2 = loadImage("5.png");

    net.setup(image1.shape);

    net.preprocess(image2);
    net.preprocess(image1);

    net.predict(image2);
    std::cout <<"Predictions\n" << net.layers[net.layers.size()-1]->output << std::endl;

    net.predict(image1);
    std::cout <<"Predictions\n" << net.layers[net.layers.size()-1]->output << std::endl;
    return 0;
}