//
// Created by Amr on 8/21/17.
//

#include "ConvLayer.h"

ConvLayer::ConvLayer(string n, int d, int c, int h, Matrix w, int nf, int hf, int wf, Matrix b, int st, int pa) : AbstractLayer(name, w, b){
    D = d;
    C = c;
    H = h;
    NF = nf;
    HF = hf;
    WF = wf;
    stride = st;
    pad = pa;
}


void ConvLayer::flattenLayer() {

}
