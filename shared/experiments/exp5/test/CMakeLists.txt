cmake_minimum_required(VERSION 3.28)  # 确保版本 >= 3.27

# Tell CMake to use the compatible compiler
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_HOST_COMPILER "/data1/jyj/opt/spack/opt/spack/linux-x86_64_v4/gcc-14.2.0-2solg7ts6ugn6mrwsd5yeousamizyv3w/bin/g++")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(MatrixMultiplication LANGUAGES CXX CUDA)

# 设置 CUDA 架构（根据你的 GPU 修改）
set(CMAKE_CUDA_ARCHITECTURES 89)  # 例如 sm_50 对应 Maxwell 架构

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# 添加可执行文件
add_executable(matrix_mul test.cu)

# 链接 CUDA 运行时库
target_link_libraries(matrix_mul PRIVATE CUDA::cudart)