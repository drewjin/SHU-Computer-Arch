cmake_minimum_required(VERSION 3.18)  # 升级到支持CUDAToolkit的最低版本

# ================================================
# 编译器设置（兼容GCC 14 + CUDA）
# ================================================
if(DEFINED ENV{SPACK_GCC_ROOT})
    set(CMAKE_C_COMPILER "$ENV{SPACK_GCC_ROOT}/bin/gcc")
    set(CMAKE_CXX_COMPILER "$ENV{SPACK_GCC_ROOT}/bin/g++")
else()
    set(SPACK_GCC_ROOT "/data1/jyj/opt/spack/opt/spack/linux-x86_64_v4/gcc-14.2.0-2solg7ts6ugn6mrwsd5yeousamizyv3w")
    set(CMAKE_C_COMPILER "${SPACK_GCC_ROOT}/bin/gcc")
    set(CMAKE_CXX_COMPILER "${SPACK_GCC_ROOT}/bin/g++")
endif()

message(STATUS "Using GCC/GXX: ${CMAKE_C_COMPILER}")
set(CMAKE_CUDA_COMPILER "/data1/jyj/micromamba/envs/cu124/bin/nvcc")

project(MatMul_Benchmark LANGUAGES CXX CUDA)  # 明确启用CUDA语言

# ================================================
# 通用编译选项
# ================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 导出编译命令（用于clangd等工具）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 构建类型配置
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "No build type specified, defaulting to Release")
endif()

# ================================================
# 依赖库配置（OpenBLAS + FMT）
# ================================================
set(CUSTOM_LIB_ROOT "/data1/jyj/Learning/SHU-Computer-Arch/third_party")

# OpenBLAS
set(OpenBLAS_INCLUDE_DIRS "${CUSTOM_LIB_ROOT}/include")
set(OpenBLAS_LIBRARIES "${CUSTOM_LIB_ROOT}/lib/libopenblas.a")
message(STATUS "Found OpenBLAS: ${OpenBLAS_INCLUDE_DIRS} | ${OpenBLAS_LIBRARIES}")

# FMT
set(FMT_INCLUDE_DIR "${CUSTOM_LIB_ROOT}/include")
set(FMT_LIBRARY "${CUSTOM_LIB_ROOT}/lib/libfmt.a")
message(STATUS "Found FMT: ${FMT_INCLUDE_DIR} | ${FMT_LIBRARY}")

# ================================================
# CUDA配置
# ================================================
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)  # 使用现代CMake的CUDAToolkit模块

# CUDA编译选项
set(CUDA_ARCH "sm_89" CACHE STRING "Target GPU architecture (e.g., sm_89)")
mark_as_advanced(CUDA_ARCH)

set(CMAKE_CUDA_ARCHITECTURES "89")

message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Host Compiler: ${CMAKE_CUDA_HOST_COMPILER}")

# ================================================
# OpenMP
# ================================================

find_package(OpenMP REQUIRED)

# ================================================
# 构建类型配置（改进版）
# ================================================
# 通用编译选项（适用于所有构建类型）
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>"
)

# 针对不同构建类型设置目标选项
function(set_target_options target)
    # C++选项
    target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-mavx;-mfma>"
        "$<$<CONFIG:Debug>:-g;-O0;-fno-omit-frame-pointer>"
        "$<$<CONFIG:Release>:-O3;-march=native;-DNDEBUG>"
    )
    
    # CUDA选项
    target_compile_options(${target} PRIVATE
        "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-O0>"
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3;-use_fast_math>"
    )
endfunction()

# ================================================
# 可执行文件配置
# ================================================
add_library(CudaMatmul SHARED src/matmul.cu)  # 建议使用SHARED库
set_target_options(CudaMatmul)

target_include_directories(CudaMatmul PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${FMT_INCLUDE_DIR}
    include
)

add_executable(MatMul_Benchmark
    src/main.cpp
    src/matmul.cpp
    src/utils.cpp
)
set_target_options(MatMul_Benchmark)

target_include_directories(MatMul_Benchmark PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${FMT_INCLUDE_DIR}
    include
)

target_link_libraries(MatMul_Benchmark PRIVATE
    CudaMatmul
    ${CUDAToolkit_LIBRARIES}
    ${OpenBLAS_LIBRARIES}
    ${FMT_LIBRARY}
    OpenMP::OpenMP_CXX
    cublas
    cudart
    m
    pthread
)