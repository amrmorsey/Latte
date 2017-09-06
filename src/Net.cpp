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
        protoxt_path), weights_dir(weights_dir) {
    cout << "Creating network...\n";
    cout << "Prototxt path - " + protoxt_path + "\n";
    cout << "Weight dir - " + weights_dir + "\n";

    ifstream network_prototxt;
    std::string layer_type, layer_name;
    network_prototxt.open(protoxt_path);

    this->mean_mat = this->loadMatrix(mean_dir, "mean");
    // Get a map of the shapes of the weighted layers
    // To be passed to getWeightAndBias
    std::map<std::string, vector<int>> shapes = this->getWeightShapes();

    while (network_prototxt >> layer_type >> layer_name) {
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
            std::unique_ptr<Matrix> weights, bias;
            tie(weights, bias) = this->getWeightAndBias(layer_name, shapes);
            std::unique_ptr<AbstractLayer> ptr(
                    new FullyConnected(layer_name, num_of_outputs, std::move(weights), std::move(bias)));
            this->layers.push_back(std::move(ptr));
        } else
            std::cerr << "Parsing Error - Ignoring \"" + layer_type + "\" as it is not a supported layer" << endl;
    }
}

std::map<std::string, vector<int>> Net::getWeightShapes() {
    int dim;
    std::string layer_name;
    std::map<string, vector<int>> shapes;

    ifstream shapes_file;
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

std::tuple<std::unique_ptr<Matrix>, std::unique_ptr<Matrix>>
Net::getWeightAndBias(const std::string &layer_name, const std::map<std::string, vector<int>> &shape_map) {
    // Get shapes of weight and bias matrices
    vector<int> weight_shape = shape_map.at(layer_name);
    vector<int> bias_shape = shape_map.at(layer_name + "_bias");

    // Get the actual weights and biases
    vector<float> weights = this->extractValues(this->weights_dir + "/" + layer_name + ".ahsf");
    vector<float> bias = this->extractValues(this->weights_dir + "/" + layer_name + "_bias.ahsf");

    // Create the weight and bias matrices
    std::unique_ptr<Matrix> weights_mat(new Matrix(weights, weight_shape));
    std::unique_ptr<Matrix> bias_mat(new Matrix(bias, bias_shape));

    return std::make_tuple(std::move(weights_mat), std::move(bias_mat));
}

// Should return a pointer to the values here instead of returning by copy
vector<float> Net::extractValues(const std::string &file_path) {
    char c;
    float val;

    ifstream file;
    file.open(file_path);

    vector<float> values;

    // Loop until beginning of array (openining '[')
    while ((file >> c) && (c != '[')) {}

    // Keep reading values until closing ']' is met
    while ((file >> val >> c) && ((c == ',') || (c == ']'))) {
        values.push_back(val);
    }

    return values;
}

void Net::printLayers() {
    for (auto &&layer : this->layers)
        std::cout << layer.get()->name << endl; // layer->name is better but gives a false error in clion
}

void Net::predict(const Matrix &image) {
    Matrix out = image;
    //out.subNoSSE(mean_mat);
    for (auto &&layer : this->layers) {
        layer.get()->calculateOutput(out);//adasdasdasd
    }
    // Get top predictions code from caffe
}

Matrix Net::loadMatrix(const string &matrix_dir, const string &matrix_name) {
    vector<float> image_vec(this->extractValues(matrix_dir + "/" + matrix_name + ".ahsf"));
    vector<int> image_shape(3);

    ifstream shape_file;
    shape_file.open(matrix_dir + "/" + matrix_name + "_shape.ahsf");
    shape_file >> image_shape[0] >> image_shape[1] >> image_shape[2];

    return Matrix(image_vec, image_shape);
}

void Net::preprocess(Matrix& m) {
    m.subNoSSE(mean_mat);
}



