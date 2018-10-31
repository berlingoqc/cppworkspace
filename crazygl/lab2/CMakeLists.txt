#
cmake_minimum_required (VERSION 3.8)



# Copie les fichiers shaders vers le dossier binary
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/fragment.glsl ${CMAKE_CURRENT_SOURCE_DIR}/vertex.glsl DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )


include_directories ("${PROJECT_SOURCE_DIR}/crazygl")



# Ajoutez une source � l'ex�cutable de ce projet.
add_executable (lab2 main.cpp house_maker.cpp house_maker.h )

target_link_libraries(lab2
	crazygl
	${OPENGL_LIBRARY}
	${GLEW_LIBRARY}
	${FREEGLUT_LIBRARY}
)




