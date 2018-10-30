# CMakeList.txt : CMake project for mvlar, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Crée ma librairie 
add_library (libmvlar "ap_cmd.cpp" "ap_cmd.h")

