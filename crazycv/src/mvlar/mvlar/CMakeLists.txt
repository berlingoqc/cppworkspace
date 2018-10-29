# CMakeList.txt : CMake project for mvlar, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)


include_directories ("${PROJECT_SOURCE_DIR}/libmvlar")
# Add source to this project's executable.
add_executable (mvlar "mvlar.cpp" "mvlar.h")

# Ajoute le lien a ma librairie
target_link_libraries (mvlar LINK_PUBLIC libmvlar)

# TODO: Add tests and install targets if needed.
