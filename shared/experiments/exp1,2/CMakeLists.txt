cmake_minimum_required(VERSION 3.12)
project(GDP_Analysis LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g3 -O0)  # -g3: 最大调试信息, -O0: 禁用优化
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
endif()

# 查找MPI依赖
find_package(MPI REQUIRED)

# 查找Python和NumPy
find_package(Python COMPONENTS Development NumPy REQUIRED)

# 设置matplotlib-cpp头文件路径
set(MATPLOTLIBCPP_INCLUDE_DIR "/usr/local/include" CACHE PATH "Path to matplotlib-cpp headers")

# 添加可执行文件
add_executable(gdp_analysis 
  src/main.cpp
  src/parse_data.cpp
  src/utils.cpp
)


# 包含头文件目录
target_include_directories(gdp_analysis PRIVATE
  ${MPI_CXX_INCLUDE_PATH}
  ${Python_INCLUDE_DIRS}
  ${Python_NumPy_INCLUDE_DIRS}
  ${MATPLOTLIBCPP_INCLUDE_DIR}
  include
)

# 链接库（关键修复：移除手动添加的MPI链接选项）
target_link_libraries(gdp_analysis PRIVATE
  MPI::MPI_CXX  # 使用CMake自动识别的MPI目标
  ${Python_LIBRARIES}
  png
)