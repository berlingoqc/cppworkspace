#
cmake_minimum_required (VERSION 3.8)



# Copie les fichiers shaders vers le dossier binary
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${CMAKE_CURRENT_SOURCE_DIR}/textures ${CMAKE_CURRENT_SOURCE_DIR}/obj DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )


include_directories ("${PROJECT_SOURCE_DIR}/crazygl")



# Ajoutez une source � l'ex�cutable de ce projet.
add_executable (procedural-city main.cpp city.cpp city.h camera.h camera.cpp)

target_link_libraries(procedural-city
	crazygl
	${OPENGL_LIBRARY}
	${GL_LIBS}
)




