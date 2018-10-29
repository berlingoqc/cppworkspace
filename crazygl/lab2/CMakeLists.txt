#
cmake_minimum_required (VERSION 3.8)



include_directories ("${PROJECT_SOURCE_DIR}/crazygl")

# Ajoutez une source à l'exécutable de ce projet.
add_executable (lab2 main.cpp)
target_link_libraries(lab2
	crazygl
	${OPENGL_LIBRARY}
	${GLEW_LIBRARY}/glew32d.lib
	${FREEGLUT_LIBRARY}/freeglutd.lib
)


