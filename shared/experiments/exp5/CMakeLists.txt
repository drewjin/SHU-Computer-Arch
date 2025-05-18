cmake_minimum_required(VERSION 3.28)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

set(CMAKE_CUDA_ARCHITECTURES 86)

project(MatrixMultiplication LANGUAGES CXX CUDA)

# 不需要再手动 enable_language(CUDA)，因为 project 已经处理了
# enable_language(CUDA)  # 可以删除

find_package(CUDAToolkit REQUIRED)

add_executable(TestProgram src/main.cu)
target_link_libraries(TestProgram PRIVATE CUDA::cudart)