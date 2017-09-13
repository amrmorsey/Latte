cmake_minimum_required(VERSION 3.6)
project(InferenceEngine)

set(CMAKE_CXX_STANDARD 11)

IF (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ELSE ()
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -std=c++11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -std=c++11 -O3 -DNDEBUG -Wall -Wno-sign-compare -Wno-uninitialized -Wno-c++11-narrowing")
ENDIF ()

set(PARSING_TEST src/layers/ConvLayer.h src/Net.cpp src/Net.h src/layers/abstract_layers/AbstractLayer.h src/layers/MaxPool.h src/layers/FullyConnected.h src/layers/InputLayer.h src/MatrixAVX.h src/main.cpp src/layers/abstract_layers/AbstractPoolingLayer.h src/layers/abstract_layers/AbstractWeightedLayer.h src/layers/ReLU.h src/layers/Softmax.h src/layers/Sigmoid.h src/AlignedVector.h src/utils.h src/utils.cpp src/layers/Tanh.h)
add_executable(InferenceEngine ${PARSING_TEST})