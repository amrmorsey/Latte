#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.h"
#include "MatrixAVX.h"
#include <stdexcept>
#include <cv.hpp>

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

// Loads the matrices.
MatrixAVX loadMatrix(const std::string &matrix_dir, const std::string &matrix_name) {
    std::vector<float> image_vec(extractValues(matrix_dir + "/" + matrix_name + ".ahsf"));
    std::vector<int> image_shape(3);

    std::ifstream shape_file;
    shape_file.open(matrix_dir + "/" + matrix_name + "_shape.ahsf");
    shape_file >> image_shape[0] >> image_shape[1] >> image_shape[2];

    return MatrixAVX(image_vec, image_shape);
}

MatrixAVX loadImage(const std::string &image_path) {
    cv::Mat img;
    img = cv::imread(image_path, cv::IMREAD_UNCHANGED);

    if (img.empty()) {
        throw std::invalid_argument(image_path + " cannot be opened");
    }

    std::vector<float> pixel_container;

    if (img.channels() == 1) {
        for (int i = 0; i < img.rows; ++i) {
            auto pixel = img.ptr<unsigned char>(i); // point to first pixel in row
            for (int j = 0; j < img.cols; ++j) {
                pixel_container.push_back(pixel[j]);
            }
        }
    } else if (img.channels() == 4) {
        cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);

        // Loop over all B's
        for (int i = 0; i < img.rows; ++i) {
            auto pixel = img.ptr<cv::Vec3b>(i); // point to first pixel in row
            for (int j = 0; j < img.cols; ++j) {
                pixel_container.push_back(pixel[j][0]);
            }
        }

        // Loop over all G's
        for (int i = 0; i < img.rows; ++i) {
            auto pixel = img.ptr<cv::Vec3b>(i); // point to first pixel in row
            for (int j = 0; j < img.cols; ++j) {
                pixel_container.push_back(pixel[j][1]);
            }
        }

        // Loop over all R's
        for (int i = 0; i < img.rows; ++i) {
            auto pixel = img.ptr<cv::Vec3b>(i); // point to first pixel in row
            for (int j = 0; j < img.cols; ++j) {
                pixel_container.push_back(pixel[j][2]);
            }
        }
    } else if (img.channels() == 3) {
        // Loop over all B's
        for (int i = 0; i < img.rows; ++i) {
            auto pixel = img.ptr<cv::Vec3b>(i); // point to first pixel in row
            for (int j = 0; j < img.cols; ++j) {
                pixel_container.push_back(pixel[j][0]);
            }
        }

        // Loop over all G's
        for (int i = 0; i < img.rows; ++i) {
            auto pixel = img.ptr<cv::Vec3b>(i); // point to first pixel in row
            for (int j = 0; j < img.cols; ++j) {
                pixel_container.push_back(pixel[j][1]);
            }
        }

        // Loop over all R's
        for (int i = 0; i < img.rows; ++i) {
            auto pixel = img.ptr<cv::Vec3b>(i); // point to first pixel in row
            for (int j = 0; j < img.cols; ++j) {
                pixel_container.push_back(pixel[j][2]);
            }
        }
    } else {
        throw std::invalid_argument(image_path + " is of innappropriate channel size");
    }

    return MatrixAVX(pixel_container, {img.rows, img.cols, img.channels()});
}

// Change the image shape to make it in columns depending on the size of the filter.
void im2col(MatrixAVX &input_mat, const std::vector<int> &filterShape, MatrixAVX &out, int s, int pad) {
    int index = 0;
    int count = 0;
    int filter_size = filterShape[0] * filterShape[1] * filterShape[2];
    int kernel_size = filterShape[0];
    int tt, yy, tem1, tem2;
    int y = input_mat.shape[1], x = input_mat.shape[0], z = input_mat.shape[2];
    for (int i = 0; i < y; i = i + s) { //length y
        for (int k = 0; k < x; k = k + s) { //width x
            for (int j = 0; j < z; j++) { // depth z
                tt = k - pad + kernel_size - 1;
                yy = i - pad + kernel_size - 1;
                if (tt < x + pad && yy < x + pad) {
                    for (int l = i - pad; l < i - pad + kernel_size && l < y + pad; l++) { // for i
                        for (int m = k - pad; m < k - pad + kernel_size && m < x + pad; m++) { //for j
                            tem1 = m;
                            tem2 = l;
                            if (tem1 >= 0 && tem2 >= 0 && tem1 < x && tem2 < y) {
                                out.setElement(static_cast<unsigned int>(index), input_mat.getElement(
                                        static_cast<unsigned int>(tem1 + tem2 * x + j * x * y)));
                            }
                            index++;
                            count++;
                            if (count >= filter_size) {
                                index += 8 - (filter_size % 8);
                                count = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}