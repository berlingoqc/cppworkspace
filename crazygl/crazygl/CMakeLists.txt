# CMakeList.txt : projet CMake pour crazygl, incluez la source et définissez
# la logique spécifique au projet ici.
#
cmake_minimum_required (VERSION 3.8)




# Ajoutez une source à l'exécutable de ce projet.
add_library (crazygl ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(crazygl
	${OPENGL_LIBRARY}
	${GLEW_LIBRARY}/glew32d.lib
	${FREEGLUT_LIBRARY}/freeglutd.lib
)



# TODO: Ajoutez des tests et installez des cibles si nécessaire.
