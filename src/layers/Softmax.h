//
// Created by shadyf on 01/09/17.
//

#ifndef INFERENCEENGINE_SOFTMAX_H
#define INFERENCEENGINE_SOFTMAX_H


#include "abstract_layers/AbstractLayer.h"

class Softmax : public AbstractLayer {
public:
    explicit Softmax(string name) : AbstractLayer(name) {};

    ~Softmax() {};

    // Can this calculation be done inplace?
    void calculateOutput(Matrix &input_mat) {
        float temp = 10;    // What is this?
        vector<float> probs;
        double sum = 0;
        for (auto weight : input_mat.matrix) {
            float pr = std::exp(weight / temp);
            sum += pr;
            probs.push_back(pr);
        }
        for (auto &pr : probs) {
            pr /= sum;
        }

        // Set input mat to probs
        input_mat = Matrix(probs, input_mat.shape);
    };
};


#endif //INFERENCEENGINE_SOFTMAX_H
