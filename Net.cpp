//
// Created by Amr on 8/22/17.
//

#include <fstream>
#include <iostream>
#include "Net.h"
#include "layers/InputLayer.h"
#include "layers/ConvLayer.h"
#include "layers/MaxPool.h"

//Net::Net(const string &weight_dir) {
//    float weight;
//    char c;
//    ifstream file;
//
//    file.open("test.txt");
//
//    vector<float> test;
//
//    // Loop until beginning of array
//    while ((file >> c) && (c != '[')) {}
//
//    while ((file >> weight >> c) && ((c == ',') || (c == ']'))) {
//        test.push_back(weight);
//    }
//    //Matrix weights(test, vector<int>{test.size()});
//    //AbstractLayer l("conv3d", weights, weights);
//    return;
//}

Net::Net(const std::string &protoxt_path, const std::string &weights_dir) : prototxt_path(protoxt_path),
                                                                            weights_dir(weights_dir) {
    ifstream network_prototxt;
    std::string layer_type, layer_name;
    network_prototxt.open(protoxt_path);

    std::map<std::string, vector<int>> shapes = this->getWeightShapes();

    while (network_prototxt.good()) {
        network_prototxt >> layer_type >> layer_name;

        if (layer_type == "Input") {
            vector<int> input_dim(4);
            network_prototxt >> input_dim[0] >> input_dim[1] >> input_dim[2] >> input_dim[3];
            std::unique_ptr<AbstractLayer> ptr(new InputLayer(layer_name, input_dim));
            this->layers.push_back(std::move(ptr));
        } else if (layer_type == "Convolution") {
            int num_of_outputs, kernel_size, stride, padding;
            network_prototxt >> num_of_outputs >> kernel_size >> stride >> padding;
            std::unique_ptr<Matrix> weights, bias;
            tie(weights, bias) = this->getWeightAndBias(layer_name, shapes);
            std::unique_ptr<AbstractLayer> ptr(
                    new ConvLayer(layer_name, num_of_outputs, std::move(weights), std::move(bias), kernel_size, stride,
                                  padding));
            this->layers.push_back(std::move(ptr));
        }
        else if (layer_type == "Pooling") {
            std::string pool;
            int kernel_size, stride, padding;
            network_prototxt >> pool >> kernel_size >> stride >> padding;
//            std::unique_ptr<AbstractLayer> ptr(new MaxPoolingLayer())
        }
    }
}

std::map<std::string, vector<int>> Net::getWeightShapes() {
    std::map<string, vector<int>> shapes;
    std::string layer_name;
    int dim;
    ifstream shapes_file;
    shapes_file.open(this->weights_dir + "/shapes.ahsf");

    while (shapes_file.good()) {
        std::vector<int> shape;
        int temp;
        shapes_file >> layer_name >> dim;
        for (int i = 0; i < dim; i++) {
            shapes_file >> temp;
            shape.push_back(temp);
        }
        shapes[layer_name] = shape;
    }

    return shapes;
}

std::tuple<std::unique_ptr<Matrix>, std::unique_ptr<Matrix>>
Net::getWeightAndBias(const std::string &layer_name, const std::map<std::string, vector<int>> &shape_map) {
    vector<int> weight_shape = shape_map.at(layer_name);
    vector<int> bias_shape = shape_map.at(layer_name + "_bias");

    vector<float> weights = this->extractValues(layer_name);
    vector<float> bias = this->extractValues(layer_name + "_bias");

    std::unique_ptr<Matrix> weights_mat(new Matrix(weights, weight_shape));
    std::unique_ptr<Matrix> bias_mat(new Matrix(bias, bias_shape));

    return std::make_tuple(std::move(weights_mat), std::move(bias_mat));
}

// Should return a pointer to the values here instead of returning by copy
vector<float> Net::extractValues(const std::string &file_name) {
    char c;
    float val;

    ifstream file;
    file.open(this->weights_dir + "/" + file_name + ".ahsf");

    vector<float> values;

    // Loop until beginning of array
    while ((file >> c) && (c != '[')) {}

    while ((file >> val >> c) && ((c == ',') || (c == ']'))) {
        values.push_back(val);
    }

    return values;
}



