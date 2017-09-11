#include <fstream>
#include <iostream>
#include "Net.h"
#include "layers/InputLayer.h"
#include "layers/ConvLayer.h"
#include "layers/MaxPool.h"
#include "layers/ReLU.h"
#include "layers/Softmax.h"
#include "layers/FullyConnected.h"

Net::Net(const std::string &protoxt_path, const std::string &weights_dir, const std::string &mean_dir) : prototxt_path(
        protoxt_path), weights_dir(weights_dir), mean_mat(loadMatrix(mean_dir, "mean")) {
    std::cout << "Creating network...\n";
    std::cout << "Prototxt path - " + protoxt_path + "\n";
    std::cout << "Weight dir - " + weights_dir + "\n";

    std::ifstream network_prototxt;
    std::string layer_type, layer_name;
    network_prototxt.open(protoxt_path);

    // Get a map of the shapes of the weighted layers
    // To be passed to getWeightAndBias
    std::map<std::string, std::vector<int>> shapes = this->getWeightShapes();

    while (network_prototxt >> layer_type >> layer_name) {
        if (layer_type == "Input") {
            std::vector<int> input_dim(4);
            network_prototxt >> input_dim[0] >> input_dim[1] >> input_dim[2] >> input_dim[3];
            std::unique_ptr<AbstractLayer> ptr(new InputLayer(layer_name, input_dim));
            this->layers.push_back(std::move(ptr));
        } else if (layer_type == "Convolution") {
            int num_of_outputs, kernel_size, stride, padding;
            network_prototxt >> num_of_outputs >> kernel_size >> stride >> padding;
            std::unique_ptr<MatrixAVX> weights, bias;
            tie(weights, bias) = this->getWeightAndBias(layer_name, shapes);
            std::unique_ptr<AbstractLayer> ptr(
                    new ConvLayer(layer_name, num_of_outputs, std::move(weights), std::move(bias), kernel_size, stride,
                                  padding));
            this->layers.push_back(std::move(ptr));
        } else if (layer_type == "Pooling") {
            std::string pool;
            int kernel_size, stride, padding;
            network_prototxt >> pool >> kernel_size >> stride >> padding;
            std::unique_ptr<AbstractLayer> ptr(new MaxPool(layer_name, kernel_size, stride, padding));
            this->layers.push_back(std::move(ptr));
        } else if (layer_type == "ReLU") {
            std::unique_ptr<AbstractLayer> ptr(new ReLU(layer_name));
            this->layers.push_back(std::move(ptr));
        } else if (layer_type == "Softmax") {
            std::unique_ptr<AbstractLayer> ptr(new Softmax(layer_name));
            this->layers.push_back(std::move(ptr));
        } else if (layer_type == "InnerProduct") {
            int num_of_outputs;
            network_prototxt >> num_of_outputs;
            std::unique_ptr<MatrixAVX> weights, bias;
            tie(weights, bias) = this->getWeightAndBias(layer_name, shapes);
            std::unique_ptr<AbstractLayer> ptr(
                    new FullyConnected(layer_name, num_of_outputs, std::move(weights), std::move(bias)));
            this->layers.push_back(std::move(ptr));
        } else
            std::cerr << "Parsing Error - Ignoring \"" + layer_type + "\" as it is not a supported layer" << std::endl;
    }
}

std::map<std::string, std::vector<int>> Net::getWeightShapes() {
    int dim;
    std::string layer_name;
    std::map<std::string, std::vector<int>> shapes;

    std::ifstream shapes_file;
    shapes_file.open(this->weights_dir + "/shapes.ahsf");
    std::vector<int> shape;
    int temp;
    while (shapes_file >> layer_name >> dim) {
        shape.clear();
        for (int i = 0; i < dim; i++) {
            shapes_file >> temp;
            shape.push_back(temp);
        }
        shapes[layer_name] = shape;
    }

    return shapes;
}

std::tuple<std::unique_ptr<MatrixAVX>, std::unique_ptr<MatrixAVX>>
Net::getWeightAndBias(const std::string &layer_name, const std::map<std::string, std::vector<int>> &shape_map) {
    // Get shapes of weight and bias matrices
    std::vector<int> weight_shape = shape_map.at(layer_name);
    std::vector<int> bias_shape = shape_map.at(layer_name + "_bias");

    // Get the actual weights and biases
    std::vector<float> weights = extractValues(this->weights_dir + "/" + layer_name + ".ahsf");
    std::vector<float> bias = extractValues(this->weights_dir + "/" + layer_name + "_bias.ahsf");

    // Create the weight and bias matrices
    std::unique_ptr<MatrixAVX> weights_mat(new MatrixAVX(weights, weight_shape));
    std::unique_ptr<MatrixAVX> bias_mat(new MatrixAVX(bias, bias_shape));

    return std::make_tuple(std::move(weights_mat), std::move(bias_mat));
}


void Net::printLayers() {
    for (auto &&layer : this->layers)
        std::cout << layer.get()->name << std::endl; // layer->name is better but gives a false error in clion
}

void Net::predict(const MatrixAVX &image) {
    for (int i = 1; i < layers.size(); ++i) {
        layers[i].get()->calculateOutput(layers[i-1]->output);
    }
}

void Net::preprocess(MatrixAVX &m) {
    MatrixAVX out(m.shape);

    // Calculate average
    float average = 0;
    unsigned int j;
    for (j = 0; j < mean_mat.size; ++j) {
        average += mean_mat.getElement(j);
    }
    average /= j;

    m.sub(average, out);
    m = out;
}

void Net::setup(MatrixAVX &image) {
    MatrixAVX in_mat = image;
    for (auto &&layer : this->layers){
        layer.get()->precompute(in_mat);
        in_mat = MatrixAVX(layer->output.shape);
    }
}



