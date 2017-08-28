cmake_minimum_required(VERSION 3.6)
project(InferenceEngine)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -msse4.1")

set(SOURCE_FILES main.cpp layers/ConvLayer.cpp layers/ConvLayer.h layers/Net.cpp layers/Net.h layers/Layer.h layers/MaxpoolingLayer.cpp layers/MaxpoolingLayer.h layers/FullyConnectedLayer.cpp layers/FullyConnectedLayer.h layers/inputLayer.cpp layers/inputLayer.h layers/outputLayer.cpp layers/outputLayer.h MatrixNN.cpp MatrixNN.h VecNN.cpp VecNN.h Matrix.cpp Matrix.h parsing_test.cpp)
add_executable(InferenceEngine ${SOURCE_FILES})

set(PARSING_TEST layers/ConvLayer.cpp layers/ConvLayer.h layers/Net.cpp layers/Net.h layers/Layer.h layers/MaxpoolingLayer.cpp layers/MaxpoolingLayer.h layers/FullyConnectedLayer.cpp layers/FullyConnectedLayer.h layers/inputLayer.cpp layers/inputLayer.h layers/outputLayer.cpp layers/outputLayer.h MatrixNN.cpp MatrixNN.h VecNN.cpp VecNN.h Matrix.cpp Matrix.h parsing_test.cpp)
add_executable(PARSING_TEST ${PARSING_TEST})