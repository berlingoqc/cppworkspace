cmake_minimum_required (VERSION 3.8)


# Set la version de cpp pour 11
set(CMAKE_CXX_STANDARD 11)
# Trouve les includes dans le repertoire
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Dit a CMake d'executer moc automatiquement quand necessaire
set(CMAKE_AUTOUIC_SEARCH_PATHS ui)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)


# Trouver mes packages qt
find_package(Qt5Widgets)
find_package(Qt5Charts)
find_package(Qt5Sql)

# Trouver OpenCV
find_package( OpenCV REQUIRED)


include_directories( ${OpenCV_INCLUDE_DIRS} )


set ( SRC_FILE src/colorwidget.cpp src/confmillefeuille.cpp src/home.cpp src/imagelabel.cpp src/imagewrapper.cpp src/main.cpp src/mainwindow.cpp src/millefeuillebd.cpp src/starttracking.cpp)
set ( HED_FILE inc/colorwidget.h inc/confmillefeuille.h inc/cvheaders.h inc/home.h inc/imagelabel.h inc/imagewrapper.h inc/mainwindow.h inc/millefeuille.h inc/millefeuillebd.h inc/starttracking.h) 

# Add source to this project's executable.
add_executable (lab_image ${SRC_FILE} ${HED_FILE})
target_link_libraries(lab_image Qt5::Widgets Qt5::Charts Qt5::Sql ${OpenCV_LIBS})

