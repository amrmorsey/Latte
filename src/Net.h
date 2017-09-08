#ifndef INFERENCEENGINE_NET_H
#define INFERENCEENGINE_NET_H

#include <vector>
#include <string>
#include <memory>
#include <map>
#include "layers/abstract_layers/AbstractLayer.h"
#include "utils.h"
class Net {
private:
    const std::string prototxt_path;
    const std::string weights_dir;

    MatrixAVX mean_mat;

    std::vector<std::unique_ptr<AbstractLayer>> layers;

    std::map<std::string, std::vector<int>> getWeightShapes();

    std::tuple<std::unique_ptr<MatrixAVX>, std::unique_ptr<MatrixAVX>>
    getWeightAndBias(const std::string &layer_name, const std::map<std::string, std::vector<int>> &shape_map);

public:
    Net(const std::string &protoxt_path, const std::string &weights_dir, const std::string &mean_dir);

    ~Net() {};

    void predict(const MatrixAVX &image);

    void printLayers();

    void preprocess(MatrixAVX&);
};


#endif //INFERENCEENGINE_NET_H
