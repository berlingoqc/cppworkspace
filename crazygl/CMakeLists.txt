# CMakeList.txt : fichier projet CMake de niveau supérieur, effectuez une configuration globale
# et incluez les sous-projets ici.
#
cmake_minimum_required (VERSION 3.8)

project ("lab2")


find_package(OpenGL REQUIRED)

set(GLEW_INCLUDE_PATH "" CACHE PATH "The directory that contains GL/glew.h" )
set(GLEW_LIBRARY "" CACHE FILEPATH "libglew.a or glew.lib" )

set(GLM_INCLUDE_PATH "" CACHE PATH "The directory that contains glm/glm.hpp" )

set(FREEGLUT_INCLUDE_PATH "" CACHE PATH "The directory that contains GL/freeglut.h" )
set(FREEGLUT_LIBRARY "" CACHE PATH "The directory that contains freeglut.lib" )

set(STB_INCLUDE_PATH "" CACHE PATH "The directory that contains stb.h")





include_directories(
	${GLEW_INCLUDE_PATH}
	${STB_INCLUDE_PATH}
	${FREEGLUT_INCLUDE_PATH}
	${GLM_INCLUDE_PATH}
)


# Incluez les sous-projets.
add_subdirectory ("crazygl")
add_subdirectory ("lab2")


