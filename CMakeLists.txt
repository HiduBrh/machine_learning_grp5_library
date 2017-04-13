cmake_minimum_required(VERSION 3.6)
project(machine_learning_grp5_library)

set(CMAKE_CXX_STANDARD 11)

include_directories("/Users/idu/Documents/ESGI/machine learning/Eigen/")

set(SOURCE_FILES library.cpp library.h)
add_library(machine_learning_grp5_library SHARED ${SOURCE_FILES})