cmake_minimum_required(VERSION 3.6)
project(InferenceEngine)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -msse4.1 -mavx -std=c++11")

set(SOURCE_FILES main.cpp layers/ConvLayer.h Net.cpp Net.h layers/abstract_layers/AbstractLayer.h layers/MaxPool.h layers/FullyConnected.h layers/InputLayer.h MatrixNN.cpp MatrixNN.h VecNN.cpp VecNN.h Matrix.cpp Matrix.h layers/abstract_layers/AbstractPoolingLayer.h VecAVX.cpp VecAVX.h layers/abstract_layers/AbstractWeightedLayer.h layers/ReLU.h layers/Softmax.h)
add_executable(InferenceEngine ${SOURCE_FILES})

set(PARSING_TEST layers/ConvLayer.h Net.cpp Net.h layers/abstract_layers/AbstractLayer.h layers/MaxPool.h layers/FullyConnected.h layers/InputLayer.h MatrixNN.cpp MatrixNN.h VecNN.cpp VecNN.h Matrix.cpp Matrix.h parsing_test.cpp layers/abstract_layers/AbstractPoolingLayer.h layers/abstract_layers/AbstractWeightedLayer.h VecAVX.cpp VecAVX.h layers/ReLU.h layers/Softmax.h)
add_executable(PARSING_TEST ${PARSING_TEST})