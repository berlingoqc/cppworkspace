cmake_minimum_required (VERSION 3.8)


# Set la version de cpp pour 11
set(CMAKE_CXX_STANDARD 11)
# Trouve les includes dans le repertoire
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Dit a CMake d'executer moc automatiquement quand necessaire
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)


# Trouver mes packages qt
find_package(Qt5Widgets)
find_package(Qt5Charts)
find_package(Qt5Sql)

# Trouver OpenCV
find_package( OpenCV REQUIRED)


include_directories( ${OpenCV_INCLUDE_DIRS} )


set ( SRC_FILE confmillefeuille.cpp main.cpp millefeuillebd.cpp)
set ( HED_FILE confmillefeuille.h millefeuille.h millefeuillebd.h) 

# Add source to this project's executable.
add_executable (lab_millefeuille ${SRC_FILE} ${HED_FILE} cvheaders.h imagewrapper.h imagewrapper.cpp)
target_link_libraries(lab_millefeuille Qt5::Widgets Qt5::Charts Qt5::Sql ${OpenCV_LIBS})

