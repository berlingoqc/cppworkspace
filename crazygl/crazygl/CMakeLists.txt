# CMakeList.txt : projet CMake pour crazygl, incluez la source et définissez
# la logique spécifique au projet ici.
#
cmake_minimum_required (VERSION 3.8)

SET(HEADER_FILES engine.h header.h shaders.h textures.h mesh.h)
SET(SOURCE_FILES engine.cpp shaders.cpp textures.cpp mesh.cpp)


# Ajoutez une source à l'exécutable de ce projet.
add_library (crazygl ${HEADER_FILES} ${SOURCE_FILES})



# TODO: Ajoutez des tests et installez des cibles si nécessaire.
