cmake_minimum_required(VERSION 3.6)
project(InferenceEngine)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -std=c++11 -O3 -DNDEBUG -fPIC -Wall -Wno-sign-compare -Wno-uninitialized")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize-m64 -mavx -msse4.1")

set(SOURCE_FILES src/main.cpp src/layers/ConvLayer.h src/Net.cpp src/Net.h src/layers/abstract_layers/AbstractLayer.h src/layers/MaxPool.h src/layers/FullyConnected.h src/layers/InputLayer.h src/old_vec/VecNN.cpp src/old_vec/VecNN.h src/Matrix.cpp src/Matrix.h src/layers/abstract_layers/AbstractPoolingLayer.h src/old_vec/VecAVX.cpp src/old_vec/VecAVX.h src/layers/abstract_layers/AbstractWeightedLayer.h src/layers/ReLU.h src/layers/Softmax.h src/layers/Sigmoid.h src/AlignedVector.h)
add_executable(InferenceEngine ${SOURCE_FILES})
#
set(PARSING_TEST src/layers/ConvLayer.h src/Net.cpp src/Net.h src/layers/abstract_layers/AbstractLayer.h src/layers/MaxPool.h src/layers/FullyConnected.h src/layers/InputLayer.h src/old_vec/VecNN.cpp src/old_vec/VecNN.h src/Matrix.cpp src/Matrix.h src/parsing_test.cpp src/layers/abstract_layers/AbstractPoolingLayer.h src/layers/abstract_layers/AbstractWeightedLayer.h src/old_vec/VecAVX.cpp src/old_vec/VecAVX.h src/layers/ReLU.h src/layers/Softmax.h src/layers/Sigmoid.h src/AlignedVector.h)
add_executable(PARSING_TEST ${PARSING_TEST})