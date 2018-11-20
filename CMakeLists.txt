cmake_minimum_required(VERSION 2.8)
#set(CMAKE_CXX_COMPILER "icpc")
project(iS3D)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_HOME_DIRECTORY}")

# Require GSL for this project:
find_package(GSL REQUIRED)

#g++ flags
set(CMAKE_CXX_FLAGS "-O3 -fopenmp -std=c++11 ")

#intel flags
#set(CMAKE_CXX_FLAGS "-O3 -qopenmp -std=c++11 ")

include_directories(${GSL_INCLUDE_DIR})

#header files
include_directories(src/cpp)

add_subdirectory (src/cpp)
