# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("crazycv")

# Include sub-projects.
add_subdirectory ("crazycv")
add_subdirectory ("lab_image")
