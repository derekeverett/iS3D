cmake_minimum_required(VERSION 2.8)
project(deltaf_table)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_HOME_DIRECTORY}")

# Require GSL for this project:
find_package(GSL REQUIRED)

#flags
set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -Wuninitialized")

include_directories(${GSL_INCLUDE_DIR})

#header files
include_directories(src)

add_subdirectory (src)
