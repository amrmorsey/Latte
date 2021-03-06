//
// Created by shadyf on 08/09/17.
//

#ifndef INFERENCEENGINE_UTILS_H
#define INFERENCEENGINE_UTILS_H

#include <string>
#include <vector>
#include <fstream>
#include "MatrixAVX.h"

std::vector<float> extractValues(const std::string &file_path);

MatrixAVX loadMatrix(const std::string &matrix_dir, const std::string &matrix_name);

MatrixAVX loadImage(const std::string &image_path);
void im2col(MatrixAVX &input_mat, const std::vector<int> &filterShape, MatrixAVX &out, int s, int pad);

#endif //INFERENCEENGINE_UTILS_H
