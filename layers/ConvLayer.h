//
// Created by Amr on 8/21/17.
//

#ifndef INFERENCEENGINE_CONVLAYER_H
#define INFERENCEENGINE_CONVLAYER_H

#include <string>
#include "Layer.h"
using namespace std;

class ConvLayer : public Layer{

private: string name;
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
    ConvLayer(string n, int d, int c, int h, int w, int nf, int hf, int wf, int b, int st, int pa);
    ~ConvLayer();
    void loadWieghts();
    void forwardPass();
    void flattenLayer();
    void feedForward();
    void setInput(double *);
};


#endif //INFERENCEENGINE_CONVLAYER_H
