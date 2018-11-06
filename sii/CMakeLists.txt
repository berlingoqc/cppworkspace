cmake_minimum_required (VERSION 3.8)

project(sii)

find_package( OpenCV REQUIRED )

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory (libcuda)
add_subdirectory (labastar)
#add_subdirectory (labsobel)


