cmake_minimum_required(VERSION 3.6)
project(InferenceEngine)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -msse4.1 -mavx")

set(SOURCE_FILES main.cpp layers/ConvLayer.cpp layers/ConvLayer.h Net.cpp Net.h layers/abstract_layers/AbstractLayer.h layers/MaxPoolingLayer.cpp layers/MaxPoolingLayer.h layers/FullyConnectedLayer.cpp layers/FullyConnectedLayer.h layers/InputLayer.cpp layers/InputLayer.h layers/OutputLayer.cpp layers/OutputLayer.h MatrixNN.cpp MatrixNN.h VecNN.cpp VecNN.h Matrix.cpp Matrix.h layers/abstract_layers/AbstractPoolingLayer.cpp layers/abstract_layers/AbstractPoolingLayer.h VecAVX.cpp VecAVX.h layers/abstract_layers/AbstractWeightedLayer.cpp layers/abstract_layers/AbstractWeightedLayer.h)
add_executable(InferenceEngine ${SOURCE_FILES})

set(PARSING_TEST layers/ConvLayer.cpp layers/ConvLayer.h Net.cpp Net.h layers/abstract_layers/AbstractLayer.h layers/MaxPoolingLayer.cpp layers/MaxPoolingLayer.h layers/FullyConnectedLayer.cpp layers/FullyConnectedLayer.h layers/InputLayer.cpp layers/InputLayer.h layers/OutputLayer.cpp layers/OutputLayer.h MatrixNN.cpp MatrixNN.h VecNN.cpp VecNN.h Matrix.cpp Matrix.h parsing_test.cpp layers/abstract_layers/AbstractPoolingLayer.cpp layers/abstract_layers/AbstractPoolingLayer.h layers/abstract_layers/AbstractWeightedLayer.cpp layers/abstract_layers/AbstractWeightedLayer.h)
add_executable(PARSING_TEST ${PARSING_TEST})