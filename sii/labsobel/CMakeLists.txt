

include_directories("${PROJECT_SOURCE_DIR}/libcuda")


add_executable(labsobel main.cpp)
target_link_libraries(labsobel
    libcuda    
    ${OpenCV_LIBS}
)