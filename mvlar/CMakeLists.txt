# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("mvlar")

set(CMAKE_CXX_STANDARD 11)

enable_testing()


SET(Boost_USE_MULTITHREAD ON)
SET(Boost_USE_STATIC_LIBS ON)
set(Boost_DEBUG ON)


find_package(Boost REQUIRED COMPONENTS system thread system atomic log)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
endif()

link_directories(${Boost_LIBRARY_DIRS})


# Include sub-projects.

add_subdirectory("libmvlar")
add_subdirectory("libcontrol")
add_subdirectory ("mvlar")

