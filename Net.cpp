//
// Created by Amr on 8/22/17.
//

#include <fstream>
#include <iostream>
#include "Net.h"

Net::Net(std::string weight_dir) {
    float weight;
    char c;
    ifstream file;

    file.open("test.txt");

    vector<float> test;

    // Loop until beginning of array
    while ((file >> c) && (c != '[')) {}

    while ((file >> weight >> c) && ((c == ',') || (c == ']'))) {
        test.push_back(weight);
    }

    return;
}
