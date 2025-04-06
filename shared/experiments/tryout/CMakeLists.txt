cmake_minimum_required(VERSION 3.10)
project(MPI_Matplotlib_Example LANGUAGES CXX)

# 启用 C++11
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 MPI
find_package(MPI REQUIRED)

# 查找 Python 和 matplotlib-cpp 依赖
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 添加可执行文件
add_executable(mpi_plot main.cpp)

# 链接 MPI 库
target_link_libraries(mpi_plot PRIVATE MPI::MPI_CXX)

# 配置 matplotlib-cpp
target_include_directories(mpi_plot PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(mpi_plot PRIVATE Python3::Python)

# 检查 NumPy 是否可用（可选）
find_package(Python3 COMPONENTS NumPy)
if(Python3_NumPy_FOUND)
    target_link_libraries(mpi_plot PRIVATE Python3::NumPy)
else()
    target_compile_definitions(mpi_plot PRIVATE WITHOUT_NUMPY)
endif()