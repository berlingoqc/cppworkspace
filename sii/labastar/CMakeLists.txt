
include_directories(${PROJECT_SOURCE_DIR}/libcuda)

link_directories(${PROJECT_SOURCE_DIR}/extra/lib/)

add_executable(labastar main.cpp 2dhelp.cpp 2dhelp.h node.cpp node.h )
if(WITH_AXIS_COM)
	include_directories(${PROJECT_SOURCE_DIR}/extra/include)
    target_link_libraries(labastar AxisCommunication)
	add_definitions(-D_WITH_AXIS_COM)
endif()

target_link_libraries(labastar
	#libcuda
    ${OpenCV_LIBS}
)
