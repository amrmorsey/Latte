//
// Created by Amr on 8/22/17.
//

#ifndef INFERENCEENGINE_NET_H
#define INFERENCEENGINE_NET_H

#include <vector>
#include <string>
#include "layers/abstract_layers/AbstractLayer.h"

class Net {
private:
    vector<AbstractLayer> layers;
    vector<AbstractLayer> bias;
public:
    Net(const string &weight_dir);

    ~Net() {}


};


#endif //INFERENCEENGINE_NET_H
