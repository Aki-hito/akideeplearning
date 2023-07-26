cmake_minimum_required(VERSION 3.25)
project(eigenMartix)

set(CMAKE_CXX_STANDARD 17)

add_executable(eigenMartix main.cpp)

#添加头文件
include_directories("/usr/include/eigen3")