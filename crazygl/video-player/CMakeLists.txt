#
cmake_minimum_required (VERSION 3.8)

file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

include_directories (${PROJECT_SOURCE_DIR}/imgui)
include_directories (${PROJECT_SOURCE_DIR}/crazygl)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Ajoutez une source � l'ex�cutable de ce projet.
add_executable (video-player main.cpp rgb_player.h )

target_link_libraries(video-player
	imgui
	crazygl
	${OpenCV_LIBS}
	${OPENGL_LIBRARY}
	${GL_LIBS}
)




