cmake_minimum_required (VERSION 3.8)

project(sii LANGUAGES CXX) # Enable c++ et cuda ( doivent être présent pour compiler l'application )


if(NOT DEFINED CMAKE_CUDA_STANDARD) # Mets le standard de c++11 pour le code compiler avec cuda
    set(CMAKE_CUDA_STANDARD 11) 
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()


# Repertoire du tclap obligatoire
set( TCLAP_INCLUDE_DIR "" CACHE PATH "The diretory containing tclap/CmdLine.h" ) # Cache path c'est pour les variables qui arrivent du mode cmd

option ( WITH_CUDA "Add cuda support and build application that required cuda" OFF)

if(WIN32)
    option( WITH_AXIS_COM "Add Axis Communication library" OFF)
endif(WIN32)

# Repertoire de axis communication optionel
if(WITH_AXIS_COM)
    #set( AXIS_COM_INCLUDE_DIR "" CACHE PATH "The directory containing AxisCommunication.h ")
    #set( AXIS_COM_LIBRARY_DIR "" CACHE PATH "The directory containing AxisCommunication.lib")
endif(WITH_AXIS_COM)




# Opencv obligatoire
find_package( OpenCV REQUIRED )

# Ajout les includes que tout le monde a besoin
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${AXIS_COM_INCLUDE_DIR}
	${TCLAP_INCLUDE_DIR}
)

if (WITH_CUDA)
    enable_language(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	add_definitions(-D_WITH_CUDA)
	set( LIBCUDA_LIBS libcuda)
    add_subdirectory (libcuda)
    add_subdirectory (labsobel)
endif()

add_subdirectory (labastar)


