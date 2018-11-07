
include_directories(${PROJECT_SOURCE_DIR}/libcuda)

link_directories(${PROJECT_SOURCE_DIR}/extra/lib/)

add_executable(labastar main.cpp 2dhelp.cpp 2dhelp.h node.cpp node.h object.h object.cpp )
if(WITH_AXIS_COM)
    target_link_libraries(labastar ${AXIS_COM_LIBRARY_DIR}/AxisCommunication.lib)
endif()

target_link_libraries(labastar
    ${OpenCV_LIBS}
)
