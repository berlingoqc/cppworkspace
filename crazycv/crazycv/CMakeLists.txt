# CMakeList.txt : CMake project for crazycv, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)


find_package(Qt5Core)
set(CMAKE_AUTOMOC ON)

set ( SOURCE src/imagewrapper.cpp )
set ( HEADER include/cvheaders.h include/imagewrapper.h )
# Add source to this project's executable.
add_library( crazycv  ${SOURCE} ${HEADER})

# TODO: Add tests and install targets if needed.
