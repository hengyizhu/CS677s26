cmake_minimum_required(VERSION 3.18)
project(CS677s26 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(cuda_hello
    src/main.cu
    src/boosting_core.cu
    src/face_vision_engine.cu
    src/model_io.cpp
)

target_include_directories(cuda_hello PRIVATE include)
set_target_properties(cuda_hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(cuda_hello PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
