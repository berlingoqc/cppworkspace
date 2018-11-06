
include_directories(${PROJECT_SOURCE_DIR}/extra/include)
#include_directories(${PROJECT_SOURCE_DIR}/libcuda)

link_directories(${PROJECT_SOURCE_DIR}/extra/lib/)

add_executable(labastar main.cpp 2dhelp.cpp 2dhelp.h node.cpp node.h object.h)
target_link_libraries(labastar
    AxisCommunication.lib
    ${OpenCV_LIBS}
)
