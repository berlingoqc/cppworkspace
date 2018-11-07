cmake_minimum_required (VERSION 3.8)

project(sii LANGUAGES CXX CUDA )


if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()


set( TCLAP_INCLUDE_DIR "" CACHE PATH "The diretory containing tclap/CmdLine.h" )

find_package( OpenCV REQUIRED )

include_directories(
    ${OpenCV_INCLUDE_DIRS}
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${TCLAP_INCLUDE_DIR}
)

add_subdirectory (libcuda)
add_subdirectory (labastar)
add_subdirectory (labsobel)


