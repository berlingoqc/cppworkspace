enable_language(CUDA)


add_library(libcuda STATIC gpusobel.h headers.h helpme.h hsvtorgbkernel.cu )

target_compile_features(libcuda PUBLIC cxx_std_11)


