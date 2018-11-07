

add_library(libcuda STATIC gpusobel.h headers.h helpme.h gpusobel.cpp hsvtorgbkernel.cu)

target_compile_features(libcuda PUBLIC cxx_std_11)

set_target_properties(libcuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
