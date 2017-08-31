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
    const string &prototxt_path;
    const string &weights_dir;
    vector<AbstractLayer> layers;
    vector<AbstractLayer> bias;

public:
    Net(const string &protoxt_path, const string &weights_dir);

    ~Net() {}


};


#endif //INFERENCEENGINE_NET_H
