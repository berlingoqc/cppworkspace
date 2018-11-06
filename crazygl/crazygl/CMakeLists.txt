# CMakeList.txt : projet CMake pour crazygl, incluez la source et définissez
# la logique spécifique au projet ici.
#
cmake_minimum_required (VERSION 3.8)

SET(HEADER_FILES "${STB_INCLUDE_PATH}/stb_image.h" engine.h header.h shaders.h mesh.h)
SET(SOURCE_FILES engine.cpp shaders.cpp mesh.cpp)


# Ajoutez une source à l'exécutable de ce projet.
add_library (crazygl ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(crazygl
	${OPENGL_LIBRARY}
	${GLEW_LIBRARY}/glew32d.lib
	${FREEGLUT_LIBRARY}/freeglutd.lib
)



# TODO: Ajoutez des tests et installez des cibles si nécessaire.
