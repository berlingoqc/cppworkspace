# CMakeList.txt : CMake project for mvlar, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

set ( HEADERS ap_cmd.h config.h control_states.h math.h navdata_common.h navdata_client.h typedef.h thread.h )
set ( SOURCES ap_cmd.cpp navdata_client.cpp )


# Crée ma librairie 
add_library (libmvlar STATIC ${HEADERS} ${SOURCES})