#
cmake_minimum_required (VERSION 3.8)



# Copie les fichiers shaders vers le dossier binary
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${CMAKE_CURRENT_SOURCE_DIR}/textures ${CMAKE_CURRENT_SOURCE_DIR}/model DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

include_directories ("${PROJECT_SOURCE_DIR}/crazygl")
include_directories ("${PROJECT_SOURCE_DIR}/imgui")

# Ajoutez une source � l'ex�cutable de ce projet.
add_executable (tiefighter main.cpp city.cpp city.h camera.h camera.cpp)

target_link_libraries(tiefighter
	crazygl
	imgui
	${OPENGL_LIBRARY}
	${GL_LIBS}
)




