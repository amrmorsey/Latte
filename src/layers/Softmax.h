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
//        float temp = 10;    // What is this?
//        vector<float> probs;
//        double sum = 0;
//        for (auto weight : input_mat.matrix) {
//            float pr = std::exp(weight / temp);
//            sum += pr;
//            probs.push_back(pr);
//        }
//        for (auto &pr : probs) {
//            pr /= sum;
//        }
//
//        // Set input mat to probs
//        input_mat = Matrix(probs, input_mat.shape);
        softMaxFunction3(input_mat);
        //output.matrix = input_mat.matrix;
    };

    void softMaxFunction(Matrix &input_mat){
        vector <float> y = input_mat.matrix;
        float ymax = *std::max_element(y.begin(), y.end());
        float sumofelements = 0;
        for (auto& n : y)
            sumofelements += n;
        for(int f = 0; f < y.size(); f++) {
            y.at(f) = exp(y.at(f) - ymax);
            float ysum = 0;
            for (int i = 0; i <=f ; i++) {
                ysum +=y.at(i);
            }
            y.at(f) = y.at(f) / ysum;
        }
        input_mat.matrix = y;
    }

    void softMaxFunction2(Matrix &input_mat){
        vector<float> y = input_mat.matrix;
        for (int i = 0; i < y.size(); i++) {
            float sum = 0;
            float ymax = *std::max_element(y.begin(), y.begin()+i);
            for (int j = 0; j <=i ; j++) {
                sum += exp(y[j] - ymax);
            }
            y[i] = exp(input_mat.matrix[i] - ymax - log(sum));
        }
        input_mat.matrix = y;
    }

    void softMaxFunction3(Matrix &input_mat){
        float sum = 0;
        for (int i = 0; i < input_mat.matrix.size(); ++i) {
            output.matrix[i] = exp(input_mat.matrix[i]);
            sum+= output.matrix[i];
        }

        for (int i = 0; i < output.matrix.size(); ++i) {
            output.matrix[i] /= sum;
        }
    }

    void precompute(Matrix& a){
        output = Matrix(a.shape);
    }
};


#endif //INFERENCEENGINE_SOFTMAX_H
