//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_NET_H
#define INFERENCEENGINE_NET_H

#include <vector>
#include <string>
#include "layers/Layer.h"

class Net {
private:
    vector<Layer> layers;
    vector<Layer> bias;
public:
    Net(std::string weight_dir);

    ~Net() {}


};


#endif //INFERENCEENGINE_NET_H
