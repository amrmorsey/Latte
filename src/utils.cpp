#include "utils.h"

// Should return a pointer to the values here instead of returning by copy
std::vector<float> extractValues(const std::string &file_path) {
    char c;
    float val;

    std::ifstream file;
    file.open(file_path);

    std::vector<float> values;

    // Loop until beginning of array (openining '[')
    while ((file >> c) && (c != '[')) {}

    // Keep reading values until closing ']' is met
    while ((file >> val >> c) && ((c == ',') || (c == ']'))) {
        values.push_back(val);
    }

    return values;
}

MatrixAVX loadMatrix(const std::string &matrix_dir, const std::string &matrix_name) {
    std::vector<float> image_vec(extractValues(matrix_dir + "/" + matrix_name + ".ahsf"));
    std::vector<int> image_shape(3);

    std::ifstream shape_file;
    shape_file.open(matrix_dir + "/" + matrix_name + "_shape.ahsf");
    shape_file >> image_shape[0] >> image_shape[1] >> image_shape[2];

    return MatrixAVX(image_vec, image_shape);
}

void im2col(MatrixAVX &input_mat, const std::vector<int> &filterShape, MatrixAVX &out, int s, int pad, int x) {
    int index = 0;

    for (int i = 0; i < input_mat.shape[1]; i = i + s) { //length y
        for (int k = 0; k < input_mat.shape[0]; k = k + s) { //width x
            for (int j = 0; j < input_mat.shape.at(2); j++) { // depth z
                int tt = k - pad + filterShape.at(0) - 1;
                int yy = i - pad + filterShape.at(0) - 1;
                if (tt < input_mat.shape[0] + pad && yy < input_mat.shape[0] + pad) {
                    for (int l = i - pad;
                         l < i - pad + filterShape.at(0) && l < input_mat.shape[1] + pad; l++) { // for i
                        for (int m = k - pad;
                             m < k - pad + filterShape.at(0) && m < input_mat.shape[0] + pad; m++) { //for j
                            int tem1 = m;
                            int tem2 = l;
                            if (tem1 < 0 || tem2 < 0 || tem1 >= input_mat.shape[0] || tem2 >= input_mat.shape[1]) {
                                index++;
                            } else {
//                                vector<int> in = {tem1, tem2, j};
                                //sdasdasll.push_back(in);
                                out.setElement(index, input_mat.getElement(
                                        tem1 + tem2 * input_mat.shape[0] + j * input_mat.shape[0] * input_mat.shape[1]));
                                //ll.push_back(this->at(in));
                                index++;
                            }

                        }
                    }
                }
            }
        }
    }
}