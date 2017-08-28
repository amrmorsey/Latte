//
// Created by Amr on 8/21/17.
//

#include "ConvLayer.h"

ConvLayer::ConvLayer(string n, int d, int c, int h, int w, int nf, int hf, int wf, int b, int st, int pa) {
    name = n;
    D = d;
    C = c;
    H = h;
    W = w;
    NF = nf;
    HF = hf;
    WF = wf;
    bias = b;
    stride = st;
    pad = pa;
}

ConvLayer::~ConvLayer() {

}

void ConvLayer::flattenLayer() {

}
