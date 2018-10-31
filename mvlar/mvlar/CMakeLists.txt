# CMakeList.txt : CMake project for mvlar, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

if(MINGW)
    set(MINGW_LIBRARIES ws2_32)
endif()

include_directories ("${PROJECT_SOURCE_DIR}/libmvlar")

# Add source to this project's executable.
add_executable (mvlar "mvlar.cpp" "mvlar.h")

# Ajoute le lien a ma librairie
target_link_libraries (mvlar libmvlar ${Boost_LIBRARIES} Boost::log_setup Boost::log ${MINGW_LIBRARIES})
target_include_directories(mvlar PUBLIC ${BOOST_INCLUDEDIR})

set (CMAKE_CXX_FLAGS "-g -Wall -DBOOST_LOG_DYN_LINK")

