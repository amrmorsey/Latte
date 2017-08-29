//
// Created by Amr on 8/21/17.
//

#ifndef INFERENCEENGINE_CONVLAYER_H
#define INFERENCEENGINE_CONVLAYER_H

#include <string>
#include "abstract_layers/AbstractLayer.h"

using namespace std;

class ConvLayer : public AbstractLayer {

private:
    string name;
    int D;
    int C;
    int H;
    int W;
    int NF;
    int HF;
    int WF;
    int *flattenedMatrix;
    int bias;
    int stride;
    int pad;
    double *input;
    double *weights;
    double *output;
    vector<int> input_dims;
    vector<int> output_dims;

public:
    ConvLayer(string n, int d, int c, int h, Matrix w, int nf, int hf, int wf, Matrix b, int st, int pa);

    ~ConvLayer() {};

    void loadWieghts();

    void forwardPass();

    void flattenLayer();

    void feedForward();

    void setInput(double *);
};


#endif //INFERENCEENGINE_CONVLAYER_H
