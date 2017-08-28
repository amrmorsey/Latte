//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_NET_H
#define INFERENCEENGINE_NET_H

#include <vector>
#include "Layer.h"
using namespace std;
class Net {

private:
    vector<Layer> layers;
public:
    Net();
    ~Net();


};


#endif //INFERENCEENGINE_NET_H
