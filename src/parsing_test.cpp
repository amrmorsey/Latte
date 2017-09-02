#include "Net.h"

using namespace std;


int main() {
    Net net("simple_what.ahsf", "weights", "mean");
    Matrix image = net.loadMatrix("image", "image");
    net.predict(image);
    return 0;
}
