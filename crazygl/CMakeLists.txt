﻿# CMakeList.txt : fichier projet CMake de niveau supérieur, effectuez une configuration globale
# et incluez les sous-projets ici.
#
cmake_minimum_required (VERSION 3.8)

project ("crazygl")

set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)

set(GLEW_INCLUDE_PATH "" CACHE PATH "The directory that contains GL/glew.h" )
set(GLEW_LIBRARY "" CACHE FILEPATH "libglew.a or glew.lib" )

set(GLM_INCLUDE_PATH "" CACHE PATH "The directory that contains glm/glm.hpp" )

set(GLFW_INCLUDE_PATH "" CACHE PATH "The directory that contains GLFW/glfw3.h" )
set(GLFW_LIBRARY "" CACHE PATH "The directory that contains glfw3.lib" )

set(GLAD_INCLUDE_PATH "" CACHE PATH "The directory that contains glad/glad.h" )
set(GLAD_LIBRARY "" CACHE PATH "The directory that contains glad.lib" )

set(SOIL_INCLUDE_PATH "" CACHE PATH "The directory that contains SOIL.h" )
set(SOIL_LIBRARY "" CACHE PATH "The directory that contains SOIL.lib" )

set(ASSIMP_INCLUDE_PATH "" CACHE PATH "The directory that contains assimp/assimp.h" )
set(ASSIMP_LIBRARY "" CACHE PATH "The directory that contains assimp.lib" )

set(STB_INCLUDE_PATH "" CACHE PATH "The directory that contains stb_image.h")


if(win32)
	set( GL_LIBS glew32 glfw3 glad SOIL assimp-vc140-mt )
else(win32)
	set( GL_LIBS GL glut GLU GLEW stdc++fs )
endif(win32)


set( GL_LIBS glew32d glfw3 SOIL assimp-vc140-mt )

include_directories(
	${GLEW_INCLUDE_PATH}
	${GLFW_INCLUDE_PATH}
	${GLAD_INCLUDE_PATH}
	${GLM_INCLUDE_PATH}
	${SOIL_INCLUDE_PATH}
	${ASSIMP_INCLUDE_PATH}
	${STB_INCLUDE_PATH}
)
#add_definitions(-DWITH_STB_IMAGE)

link_directories(
	${GLEW_LIBRARY}
	${ASSIMP_LIBRARY}
	${GLFW_LIBRARY}
	${GLAD_LIBRARY}
	${SOIL_LIBRARY}
)


# Incluez les sous-projets.
add_subdirectory ("crazygl")
add_subdirectory ("imgui")
##add_subdirectory ("video-player")
add_subdirectory ("tiefighter")
add_subdirectory ("procedural-city")


