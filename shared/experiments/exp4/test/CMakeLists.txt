cmake_minimum_required(VERSION 3.20)
project(OpenMP_Example
    VERSION 1.0
    DESCRIPTION "OpenMP Example"
    LANGUAGES C
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 查找OpenMP包
find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -mavx2 -mfma -fopenmp")

# 编译器选项
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
)

# 根据不同构建类型设置选项
string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)
if(BUILD_TYPE_UPPER STREQUAL "DEBUG")
    add_compile_options(-g -O0)
else()
    add_compile_options(-O3)
endif()

# 添加可执行文件
add_executable(omp_program hellomp.c)

# 链接OpenMP并设置目标属性
target_link_libraries(omp_program 
    PUBLIC 
        OpenMP::OpenMP_C
)

# 安装规则（可选）
install(TARGETS omp_program
    RUNTIME DESTINATION bin
)

# 生成编译数据库（用于IDE支持）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 单元测试支持（可选）
enable_testing()
add_test(NAME omp_test COMMAND omp_program)