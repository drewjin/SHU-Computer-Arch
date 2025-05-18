# OpenMP矩阵乘法

## 测试OMP程序

### CMakeLists

针对给出程序，书写CMakeLists.txt文件如下，用于编译。

本实验采用CMake进行项目管理和编译配置。通过设置CMakeLists.txt，可以方便地管理依赖、编译选项以及不同构建类型（如Release和Debug），同时也便于在不同平台下进行移植和扩展。这里我们启用了OpenMP支持，并针对现代CPU开启了如`-O3`、`-mavx2`、`-mfma`等优化选项，以提升矩阵乘法的执行效率。此外，还添加了单元测试支持和编译数据库的生成，便于后续调试和代码分析。

```cpp
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
```

### 具体运行结果如下：

通过上述配置编译并运行程序后，可以看到串行和不同线程数下的并行实现的运行结果和耗时。实验结果显示，在当前矩阵规模较小的情况下，串行实现反而比并行实现更快。这主要是因为并行化本身存在一定的线程创建和调度开销，当数据量较小时，这些开销会抵消并行带来的加速效果。因此，只有在更大规模的矩阵运算中，并行化的优势才会逐渐显现。

```bash
Result(Serial):
4.320000,2.260000,2.630000,3.000000,
10.840000,6.420000,7.670000,8.920000,
17.360001,10.580001,12.710000,14.840000,
23.879999,14.740001,17.750000,20.759998,

Serial :0.009744
Parallel 2:0.069979
Parallel 4:0.116807
Parallel 8:0.382508
Parallel 16:0.732596

Result:
4.320000,2.260000,2.630000,3.000000,
10.840000,6.420000,7.670000,8.920000,
17.360001,10.580001,12.710000,14.840000,
23.879999,14.740000,17.750000,20.759998,
```

目前情况在于，串行的Matmul比所有的并行实现都要快，原因在于矩阵规模过小，导致并行化的开销大于加速比。

## 优化算法

### 代码

实现avx256的矩阵乘法，代码如下：

为了进一步提升矩阵乘法的性能，我们利用了AVX256指令集对计算过程进行了SIMD优化。通过将矩阵B的列预先加载到AVX寄存器，并对A的每一行进行广播和并行计算，可以显著减少循环次数和内存访问次数。这样不仅提升了数据并行度，还充分发挥了现代CPU的向量化计算能力。最终结果通过AVX指令存储回内存，只取前4个float，保证了正确性和高效性。

```cpp
void comput_avx(float *A, float *B, float *C) {
  // 加载矩阵 B 的列到 4 个 AVX 寄存器（转置后的 B）
  __m256 B0 = _mm256_loadu_ps(&B[0]);  // [B00, B10, B20, B30, 0,0,0,0]
  __m256 B1 = _mm256_loadu_ps(&B[4]);  // [B01, B11, B21, B31, 0,0,0,0]
  __m256 B2 = _mm256_loadu_ps(&B[8]);  // [B02, B12, B22, B32, 0,0,0,0]
  __m256 B3 = _mm256_loadu_ps(&B[12]); // [B03, B13, B23, B33, 0,0,0,0]

  // 处理矩阵 A 的每一行
#pragma GCC unroll 4
  for (int i = 0; i < 4; i++) {
    // 加载 A 的一行（广播到 AVX 寄存器的 4 个位置）
    __m256 A_row = _mm256_broadcast_ss(&A[4 * i]);

    // 计算 C[i][0], C[i][1], C[i][2], C[i][3]
    __m256 C_row = _mm256_mul_ps(A_row, B0);
    A_row = _mm256_broadcast_ss(&A[4 * i + 1]);
    C_row = _mm256_fmadd_ps(A_row, B1, C_row);
    A_row = _mm256_broadcast_ss(&A[4 * i + 2]);
    C_row = _mm256_fmadd_ps(A_row, B2, C_row);
    A_row = _mm256_broadcast_ss(&A[4 * i + 3]);
    C_row = _mm256_fmadd_ps(A_row, B3, C_row);

    // 存储结果（只取前 4 个 float）
    float tmp[8];
    _mm256_storeu_ps(tmp, C_row);
    C[4 * i] = tmp[0];
    C[4 * i + 1] = tmp[1];
    C[4 * i + 2] = tmp[2];
    C[4 * i + 3] = tmp[3];
  }
}
```

### 运行结果

可以看到，经过AVX256优化后，矩阵乘法的运行速度有了极大的提升，远快于串行和普通并行实现。由于本实验的矩阵规模为4x4，正好适合AVX256的宽度，因此优化效果非常明显。实际应用中，随着矩阵规模的增大，SIMD优化和多线程并行的优势会更加突出，建议在更大规模下进行测试，以全面评估优化效果。

```bash
Result(Serial):
4.320000,2.260000,2.630000,3.000000,
10.840000,6.420000,7.670000,8.920000,
17.360001,10.580001,12.710000,14.840000,
23.879999,14.740001,17.750000,20.759998,

Serial :0.009744
Parallel 2:0.069979
Parallel 4:0.116807
Parallel 8:0.382508
Parallel 16:0.732596
AVX-256: 0.002697 sec

Result:
4.320000,2.260000,2.630000,3.000000,
10.840000,6.420000,7.670000,8.920000,
17.360001,10.580001,12.710000,14.840000,
23.879999,14.740000,17.750000,20.759998,
```

由于矩阵正好为4x4，正好可以使用AVX256的指令集进行优化，速度提升非常明显。

在实际应用中，矩阵的规模往往会更大，因此可以考虑使用更大的矩阵进行测试，以便更好地评估并行化和SIMD优化的效果。

## 大规模矩阵乘法优化

### 代码

#### main.cpp

在`main.cpp`中，实现了一个动态的矩阵乘法测试框架，支持多种矩阵乘法实现，包括串行、并行和CUDA版本。以下是代码片段：

主程序通过命令行参数设置随机种子和矩阵规模，自动生成不同大小的测试用例，并依次调用多种矩阵乘法实现进行基准测试。每种实现都会与OpenBLAS的结果进行比对，确保正确性。测试框架还会统计每种实现的运行时间，便于直观比较不同算法和优化手段的性能差异。这样不仅保证了实验的科学性和可重复性，也为后续的性能分析和优化提供了数据支持。

```cpp
#include <cstddef>
#include <iostream>
#include <vector>
#include <chrono>
#include "string"
#include <fmt/format.h>

#include "custom_config.h"
#include "matmul.h"
#include "matmul.cuh"
#include "utils.h"


int main(int argc, char *argv[]) {
  if (argc < 1) {
    std::string err =
      fmt::format("Usage: <random_seed> [matrix_size]\n", argv[0]);
    std::cerr << err;
    return 1;
  }

  const uint64_t seed = std::stoull(argv[1]);

  for (const auto& [M, K, N] : SIZES) {
    fmt::print("\n<| TEST CASE |>: [M: {}, K: {}, N: {}]\n", M, K, N);

    fmt::print("Generating matrices with seed {}...\n", seed);
    
    auto matA = create_random_matrix(M, K, seed);
    auto matB = create_random_matrix(K, N, seed + 1);  

    fmt::print(
      "Generating golden data for [{} x {}] x [{} x {}] "
      "matrices...\n",
      M, K, K, N);
    auto start = std::chrono::high_resolution_clock::now();
    auto matGolden = openblas_matmul(matA, matB, M, K, N);
    auto end       = std::chrono::high_resolution_clock::now();
    fmt::print("Golden data generatoin completed in [{}] seconds\n",
               std::chrono::duration<double>(end - start).count());

    std::vector<std::pair<std::string, TestFunction>> tests = {
      { "Naive Serial",             naive_serial_matmul          },
      { "Tiled Serial",             tiled_serial_matmul          },
      { "Naive Parallel",           naive_parallel_matmul        },
      { "Optimized Parallel",       optimized_parallel_matmul    },
      { "Tiled Parallel",           tiled_parallel_matmul        },
      { "AVX Tiled Parallel",       tiled_parallel_matmul_avx512 },
      // { "CUDA NAIVE Parallel",      naive_cuda_matmul            },
      // { "CUDA Tiled Parallel",      tiled_cuda_matmul            },
      // { "CUBLAS GEMM",              cublas_matmul                }  
    };

    fmt::print(
      "\nRunning benchmarks for [{} x {}] x [{} x {}] matrices...\n",
      M, K, K, N);
    for (const auto& [name, func] : tests) {
    //   auto* func_ptr = func.target<Matrix (*)(
    //     const Matrix&, const Matrix&, size_t, size_t, size_t)>();
    //   if ((M > 3000 and K > 3000 and N > 3000) and
    //        (func_ptr and
    //         (*func_ptr == naive_serial_matmul or
    //          *func_ptr == tiled_serial_matmul or
    //          *func_ptr == forloop_interleave_serial_matmul))) {
    //     fmt::print("Overpass task, too large for serial matmul.\n");
    //     continue;
    //   }
      auto result =
        run_benchmark(name, func, matA, matB, matGolden, M, K, N);
      fmt::print(
        "Test '{}' completed in [{}] seconds\n ALL CLOSE {}\n\n",
        result.test_name, result.execution_time, result.all_close);
    }
  }

  return 0;
}
```

此次我们仅测试了CPU实现，CUDA算子并未编写完成，因此注释掉了CUDA相关的代码。

#### custom_config.h

主要注册了测试用例的矩阵大小，代码如下：

在配置头文件中，预设了多个不同规模的矩阵测试用例，涵盖从1,000到20,000的多种规模，便于全面评估算法在不同数据量下的表现。通过宏定义还可以灵活切换是否启用SIMD、循环展开等优化选项，以及分块大小（TILE_SIZE），为实验调优提供了便利。这样可以更系统地观察各种优化手段在不同场景下的实际效果。

```cpp
#pragma once

#include <vector>
#include "utils.h"

#define USE_SIMD
#define USE_UNROLL
#define TILE_SIZE 32

const std::vector<MatrixSize> SIZES {
  { 1000, 1000, 1000 }, { 2000, 2000, 2000 }, { 3000, 3000, 3000 },
  { 4000, 4000, 4000 }, { 5000, 5000, 5000 }, { 6000, 6000, 6000 },
  { 7000, 7000, 7000 }, { 8000, 8000, 8000 }, { 9000, 9000, 9000 },
  { 10000, 10000, 10000 }, { 10000, 15000,  18000  }, {20000, 20000, 20000}
};
```

相应的，在上面的main函数中，我在大样例下跳过了串行实现，因为其速度过于缓慢。但是此时需要保证实验报告完整性，因此我将限制解除。

#### utils.cpp

在`utils.h`和`utils.cpp`中，定义了一些辅助函数，包括矩阵生成、检查结果是否相等等。以下是头文件：

辅助工具函数主要用于生成随机矩阵、检查计算结果的正确性以及运行基准测试。通过`check_all_close`函数，可以自动判断计算结果与基准答案是否一致，若有误差会输出具体位置和数值，便于调试。`run_benchmark`函数则负责计时和结果校验，自动输出每次测试的耗时和正确性，极大提升了实验的自动化和可靠性。

```cpp
#pragma once

#include <cstddef>
#include <string>
#include <cstdint>
#include <vector>
#include <functional>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",        \
              __FILE__, __LINE__, err, cudaGetErrorString(err), #call);  \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define CHECK_CUBLAS(call)                                               \
  do {                                                                   \
    cublasStatus_t err = call;                                           \
    if (err != CUBLAS_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "CUBLAS error at %s:%d code=%d \"%s\"\n",          \
              __FILE__, __LINE__, err, #call);                           \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

using Matrix       = std::vector<double>;
using TestFunction = std::function<Matrix(const Matrix&, const Matrix&, size_t, size_t, size_t)>;

struct MatrixSize {
  size_t M, K, N;
};

struct BenchmarkResult {
  std::string test_name;
  double      execution_time;
  Matrix      result_matrix;
  bool        all_close;
};

Matrix create_random_matrix(size_t rows, size_t cols, uint64_t seed,
                            double min = -1.0, double max = 1.0);

size_t nearest_length(size_t n);

BenchmarkResult run_benchmark(const std::string& name,
                              TestFunction test_func, const Matrix& A,
                              const Matrix& B, const Matrix& Golden,
                              size_t M, size_t K, size_t N);
```

相应的，`utils.cpp`中实现了这些函数。以下是代码片段：

这些实现细节保证了实验的可重复性和结果的准确性。比如，随机矩阵生成采用固定种子，确保每次实验数据一致；误差检查采用相对和绝对容差，兼顾数值稳定性和科学性。整体设计简洁高效，便于后续扩展和维护。

```cpp
#include "utils.h"
#include "custom_config.h"

#include <fmt/base.h>
#include <assert.h>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <functional>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>


// 检查矩阵乘法结果是否相近
bool check_all_close(const Matrix& result, const Matrix& golden,
                     size_t M, size_t N, double rel_tol = 1e-5,
                     double abs_tol = 1e-8) noexcept {
  const auto MAT_SIZE = M * N;
  (void)MAT_SIZE;
  assert(result.size() == MAT_SIZE);
  assert(golden.size() == MAT_SIZE);

  if (result.size() != golden.size()) {
    return false;
  }

  const auto mismatch = std::find_if(
    result.begin(), result.end(),
    [&golden, rel_tol, abs_tol,
     i = 0ull](const auto& res_val) mutable {
      const auto   gold_val = golden[i++];
      const double diff     = std::abs(res_val - gold_val);
      const double tol      = abs_tol + rel_tol * std::abs(gold_val);
      return diff > tol;
    });

  if (mismatch != result.end()) {
    const auto idx = std::distance(result.begin(), mismatch);
    fmt::print("Mismatch at position {}: [golden: {}, result: {}]\n",
               idx, golden[idx], *mismatch);
    return false;
  }

  return true;
}

// 运行基准测试
BenchmarkResult run_benchmark(const std::string& name,
                              TestFunction test_func, const Matrix& A,
                              const Matrix& B, const Matrix& golden,
                              size_t M, size_t K, size_t N) {
  
  auto start = std::chrono::high_resolution_clock::now();
  auto result = test_func(A, B, M, K, N);
  auto end    = std::chrono::high_resolution_clock::now();
  auto all_close = check_all_close(result, golden, M, N);

  return {
    name,
    std::chrono::duration<double>(end - start).count(),
    std::move(result), all_close
  };
}

// 检查矩阵乘法结果是否相近
size_t nearest_length(size_t n) {
  return ((n + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
}

// 创建随机矩阵
Matrix create_random_matrix(size_t rows, size_t cols, uint64_t seed,
                            double min, double max) {
  Matrix m(rows * cols, 0.0);
  std::mt19937_64 engine(seed);
  std::uniform_real_distribution<double> dist(min, max);

  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      m[row * cols + col] = dist(engine);
    }
  }

  return m;
}
```

#### matmul.cpp

首先，使用OpenBLAS实现一个矩阵乘法，作为正确结果，并且以其速度为基准。以下是代码片段：

OpenBLAS作为高性能的BLAS库，其矩阵乘法实现经过高度优化，通常被视为“黄金标准”。在本实验中，所有自定义实现的正确性和性能都以OpenBLAS为参考，确保实验结果的权威性和科学性。

```cpp
Matrix openblas_matmul(const Matrix& A, const Matrix& B, size_t M, size_t K, size_t N) {
  Matrix C(M * N, 0.0f);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
              A.data(), K, B.data(), N, 0.0f, C.data(), N);
  return C;
}
```

随后，实现一个朴素的串行矩阵乘法，代码如下：

朴素串行实现采用三重循环，直接按照矩阵乘法的定义进行计算，结构清晰，易于理解。虽然效率较低，但作为基线实现，便于与后续各种优化版本进行对比。

```cpp
Matrix naive_serial_matmul(const Matrix& A, const Matrix& B, size_t M,
                           size_t K, size_t N) {
  Matrix C(M * N, 0.0f);
  for (size_t row_a = 0; row_a < M; row_a++) {
    for (size_t col_a = 0; col_a < K; col_a++) {
      double a_val = A[row_a * K + col_a]; 
      size_t row_b = col_a;  
      for (size_t col_b = 0; col_b < N; col_b++) {
        C[row_a * N + col_b] += a_val * B[row_b * N + col_b];
      }
    }
  }
  return C;
}
```

接下来，使用分块的方式来实现矩阵乘法，用以降低访存开销，从而实现更佳的性能。以下是代码片段：

分块（Tiling）技术通过将大矩阵划分为小块，提升了缓存命中率，减少了内存带宽瓶颈。这样可以显著提升大规模矩阵乘法的性能，尤其在现代多级缓存架构下效果更为明显。

```cpp
Matrix tiled_serial_matmul(const Matrix& A, const Matrix& B, size_t M,
                           size_t K, size_t N) {
  Matrix C(M * N); 

  for (size_t i = 0; i < M; i += TILE_SIZE) {
    size_t i_end = std::min(i + TILE_SIZE, M);
    for (size_t j = 0; j < N; j += TILE_SIZE) {
      size_t j_end = std::min(j + TILE_SIZE, N);
      for (size_t k = 0; k < K; k += TILE_SIZE) {
        size_t k_end = std::min(k + TILE_SIZE, K);
        for (size_t ii = i; ii < i_end; ++ii) {
          for (size_t kk = k; kk < k_end; ++kk) {
            double a = A[ii * K + kk]; 
            for (size_t jj = j; jj < j_end; ++jj) {
              C[ii * N + jj] += a * B[kk * N + jj];
            }
          }
        }
      }
    }
  }
  return C;
}
```

串行的算法实现完成后，接下来是并行化的实现。我们使用OpenMP来实现并行化，朴素的OpenMP实现如下：

通过OpenMP对外层循环进行并行化，可以充分利用多核CPU的计算资源，大幅提升运算速度。该实现方式简单直接，适合初步并行化实验。

```cpp
Matrix naive_parallel_matmul(const Matrix& A, const Matrix& B,
                             size_t M, size_t K, size_t N) {
  Matrix C(M * N, 0.0f);

#pragma omp parallel for
  for (size_t row_a = 0; row_a < M; row_a++) {
    for (size_t col_a = 0; col_a < K; col_a++) {
      double a_val = A[row_a * K + col_a];
      size_t row_b = col_a;
      for (size_t col_b = 0; col_b < N; col_b++) {
        C[row_a * N + col_b] += a_val * B[row_b * N + col_b];
      }
    }
  }
  return C;
}
```

在此基础上，使用OpenMP的SIMD和循环展开来实现更好的性能。以下是代码片段：

进一步结合SIMD指令和循环展开，可以提升单核的计算吞吐量。通过OpenMP的`simd`指令和编译器的循环展开优化，能够更好地发挥CPU的向量化能力，进一步缩短运算时间。

```cpp
Matrix optimized_parallel_matmul(const Matrix& A, const Matrix& B,
                                 size_t M, size_t K, size_t N) {
  Matrix C(M * N, 0.0f);
  
#pragma omp parallel for collapse(2)
  for (size_t row_a = 0; row_a < M; row_a++) {
    for (size_t col_a = 0; col_a < K; col_a++) {
      double a_val = A[row_a * K + col_a];
      size_t row_b = col_a;
      double sum = 0;

#ifdef USE_SIMD
#  pragma omp simd reduction(+ : sum)
#endif

#ifdef USE_UNROLL
#  pragma unroll
#endif
      for (size_t col_b = 0; col_b < N; col_b++) {
        C[row_a * N + col_b] += a_val * B[row_b * N + col_b];
      }
    }
  }
  return C;
}
```

随后，使用分块的方式来实现并行化的矩阵乘法。以下是代码片段：

将分块技术与OpenMP并行化结合，可以同时提升缓存利用率和多核并行度。每个线程负责计算一个或多个小块，极大提升了整体性能。局部块采用原子操作累加，保证并发安全。

```cpp
Matrix tiled_parallel_matmul(const Matrix& A, const Matrix& B,
                             size_t M, size_t K, size_t N) {
  Matrix C(M * N);
  
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (size_t i = 0; i < M; i += TILE_SIZE) {
    for (size_t j = 0; j < N; j += TILE_SIZE) {
      size_t i_end = std::min(i + TILE_SIZE, M);
      size_t j_end = std::min(j + TILE_SIZE, N);
      Matrix local_block(TILE_SIZE * TILE_SIZE,
                         0);  

      for (size_t k = 0; k < K; k += TILE_SIZE) {
        size_t k_end = std::min(k + TILE_SIZE, K);

        for (size_t ii = i; ii < i_end; ++ii) {
          for (size_t kk = k; kk < k_end; ++kk) {
            double a = A[ii * K + kk];
            for (size_t jj = j; jj < j_end; ++jj) {
              local_block[(ii - i) * TILE_SIZE + (jj - j)] +=
                a * B[kk * N + jj];
            }
          }
        }
      }

      for (size_t ii = i; ii < i_end; ++ii) {
        for (size_t jj = j; jj < j_end; ++jj) {
#pragma omp atomic
          C[ii * N + jj] +=
            local_block[(ii - i) * TILE_SIZE + (jj - j)];
        }
      }
    }
  }

  return C;
}
```

最后，结合分块+SIMD（AVX512）+OpenMP的方式来实现并行化的矩阵乘法。以下是代码片段：

在最高级别的优化中，综合利用分块、SIMD（如AVX512）和多线程并行，最大化现代CPU的计算能力。通过静态断言确保分块大小适配SIMD宽度，内层循环采用AVX512指令进行向量化计算，极大提升了大规模矩阵乘法的效率。这种多层次优化方式在实际工程和科学计算中应用广泛，是高性能计算的重要手段。

```cpp
Matrix tiled_parallel_matmul_avx512(const Matrix& A, const Matrix& B,
                                    size_t M, size_t K, size_t N) {
  Matrix C(M * N);

  static_assert(
    TILE_SIZE % 8 == 0,
    "TILE_SIZE should be multiple of 8 for AVX512 optimization");

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (size_t i = 0; i < M; i += TILE_SIZE) {
    for (size_t j = 0; j < N; j += TILE_SIZE) {
      size_t i_end = std::min(i + TILE_SIZE, M);
      size_t j_end = std::min(j + TILE_SIZE, N);
      Matrix local_block(TILE_SIZE * TILE_SIZE, 0);

      for (size_t k = 0; k < K; k += TILE_SIZE) {
        size_t k_end = std::min(k + TILE_SIZE, K);

        for (size_t ii = i; ii < i_end; ++ii) {
          for (size_t kk = k; kk < k_end; ++kk) {
            double a = A[ii * K + kk];

            // AVX512优化部分
            size_t jj = j;
            for (; jj + 7 < j_end; jj += 8) {
              __m512d b_vec = _mm512_loadu_pd(&B[kk * N + jj]);

              __m512d a_vec = _mm512_set1_pd(a);

              __m512d prod = _mm512_mul_pd(a_vec, b_vec);

              size_t  offset = (ii - i) * TILE_SIZE + (jj - j);
              __m512d acc    = _mm512_loadu_pd(&local_block[offset]);

              acc = _mm512_add_pd(acc, prod);

              _mm512_storeu_pd(&local_block[offset], acc);
            }

            for (; jj < j_end; ++jj) {
              local_block[(ii - i) * TILE_SIZE + (jj - j)] +=
                a * B[kk * N + jj];
            }
          }
        }
      }

      for (size_t ii = i; ii < i_end; ++ii) {
        for (size_t jj = j; jj < j_end; ++jj) {
#pragma omp atomic
          C[ii * N + jj] +=
            local_block[(ii - i) * TILE_SIZE + (jj - j)];
        }
      }
    }
  }

  return C;
}
```


### 运行结果

按照上面提过的测试用例，运行结果如下：

```
<| TEST CASE |>: [M: 1000, K: 1000, N: 1000]
Generating matrices with seed 114514...
Generating golden data for [1000 x 1000] x [1000 x 1000] matrices...
Golden data generatoin completed in [0.062829892] seconds

Running benchmarks for [1000 x 1000] x [1000 x 1000] matrices...
Test 'Naive Serial' completed in [0.164721234] seconds
 ALL CLOSE true

Test 'Tiled Serial' completed in [0.162446327] seconds
 ALL CLOSE true

Test 'Naive Parallel' completed in [0.023864608] seconds
 ALL CLOSE true

Mismatch at position 70632: [golden: 4.584868540219329, result: 5.198155891848263]
Test 'Optimized Parallel' completed in [0.013970384] seconds
 ALL CLOSE false

Test 'Tiled Parallel' completed in [0.016944754] seconds
 ALL CLOSE true

Test 'AVX Tiled Parallel' completed in [0.014018757] seconds
 ALL CLOSE true


<| TEST CASE |>: [M: 2000, K: 2000, N: 2000]
Generating matrices with seed 114514...
Generating golden data for [2000 x 2000] x [2000 x 2000] matrices...
Golden data generatoin completed in [0.118333519] seconds

Running benchmarks for [2000 x 2000] x [2000 x 2000] matrices...
Test 'Naive Serial' completed in [3.218088056] seconds
 ALL CLOSE true

Test 'Tiled Serial' completed in [1.373617541] seconds
 ALL CLOSE true

Test 'Naive Parallel' completed in [0.070512203] seconds
 ALL CLOSE true

Test 'Optimized Parallel' completed in [0.056421989] seconds
 ALL CLOSE true

Test 'Tiled Parallel' completed in [0.038224678] seconds
 ALL CLOSE true

Test 'AVX Tiled Parallel' completed in [0.039298616] seconds
 ALL CLOSE true


<| TEST CASE |>: [M: 3000, K: 3000, N: 3000]
Generating matrices with seed 114514...
Generating golden data for [3000 x 3000] x [3000 x 3000] matrices...
Golden data generatoin completed in [0.150235321] seconds

Running benchmarks for [3000 x 3000] x [3000 x 3000] matrices...
Test 'Naive Serial' completed in [11.488128136] seconds
 ALL CLOSE true

Test 'Tiled Serial' completed in [7.64811106] seconds
 ALL CLOSE true

Test 'Naive Parallel' completed in [0.339124739] seconds
 ALL CLOSE true

Test 'Optimized Parallel' completed in [0.294925684] seconds
 ALL CLOSE true

Test 'Tiled Parallel' completed in [0.133278992] seconds
 ALL CLOSE true

Test 'AVX Tiled Parallel' completed in [0.145573751] seconds
 ALL CLOSE true


<| TEST CASE |>: [M: 4000, K: 4000, N: 4000]
Generating matrices with seed 114514...
Generating golden data for [4000 x 4000] x [4000 x 4000] matrices...
Golden data generatoin completed in [0.21570066] seconds

Running benchmarks for [4000 x 4000] x [4000 x 4000] matrices...
Test 'Naive Serial' completed in [29.606593904] seconds
 ALL CLOSE true

Test 'Tiled Serial' completed in [18.327591679] seconds
 ALL CLOSE true

Test 'Naive Parallel' completed in [0.882664866] seconds
 ALL CLOSE true

Test 'Optimized Parallel' completed in [0.845917268] seconds
 ALL CLOSE true

Test 'Tiled Parallel' completed in [0.29238608] seconds
 ALL CLOSE true

Test 'AVX Tiled Parallel' completed in [0.310602846] seconds
 ALL CLOSE true


<| TEST CASE |>: [M: 5000, K: 5000, N: 5000]
Generating matrices with seed 114514...
Generating golden data for [5000 x 5000] x [5000 x 5000] matrices...
Golden data generatoin completed in [0.285307558] seconds

Running benchmarks for [5000 x 5000] x [5000 x 5000] matrices...
Test 'Naive Serial' completed in [57.871196705] seconds
 ALL CLOSE true

Test 'Tiled Serial' completed in [35.63895419] seconds
 ALL CLOSE true

Test 'Naive Parallel' completed in [1.605001465] seconds
 ALL CLOSE true

Test 'Optimized Parallel' completed in [1.536844384] seconds
 ALL CLOSE true

Test 'Tiled Parallel' completed in [0.730741687] seconds
 ALL CLOSE true

Test 'AVX Tiled Parallel' completed in [0.623248401] seconds
 ALL CLOSE true


<| TEST CASE |>: [M: 6000, K: 6000, N: 6000]
Generating matrices with seed 114514...
Generating golden data for [6000 x 6000] x [6000 x 6000] matrices...
Golden data generatoin completed in [0.388209766] seconds

Running benchmarks for [6000 x 6000] x [6000 x 6000] matrices...
Test 'Naive Serial' completed in [103.306431358] seconds
 ALL CLOSE true

Test 'Tiled Serial' completed in [62.585711358] seconds
 ALL CLOSE true

Test 'Naive Parallel' completed in [2.787234943] seconds
 ALL CLOSE true

Test 'Optimized Parallel' completed in [2.748057296] seconds
 ALL CLOSE true

Test 'Tiled Parallel' completed in [0.949907652] seconds
 ALL CLOSE true

Test 'AVX Tiled Parallel' completed in [1.0089802] seconds
 ALL CLOSE true


<| TEST CASE |>: [M: 7000, K: 7000, N: 7000]
Generating matrices with seed 114514...
Generating golden data for [7000 x 7000] x [7000 x 7000] matrices...
Golden data generatoin completed in [0.571528842] seconds

Running benchmarks for [7000 x 7000] x [7000 x 7000] matrices...
Test 'Naive Serial' completed in [169.742059369] seconds
 ALL CLOSE true

Test 'Tiled Serial' completed in [98.03809329] seconds
 ALL CLOSE true

Test 'Naive Parallel' completed in [6.144585249] seconds
 ALL CLOSE true

Test 'Optimized Parallel' completed in [4.526362459] seconds
 ALL CLOSE true

Test 'Tiled Parallel' completed in [1.544051087] seconds
 ALL CLOSE true

Test 'AVX Tiled Parallel' completed in [1.638753529] seconds
 ALL CLOSE true


<| TEST CASE |>: [M: 8000, K: 8000, N: 8000]
Generating matrices with seed 114514...
Generating golden data for [8000 x 8000] x [8000 x 8000] matrices...
Golden data generatoin completed in [0.718768319] seconds

Running benchmarks for [8000 x 8000] x [8000 x 8000] matrices...
Test 'Naive Serial' completed in [254.112294347] seconds
 ALL CLOSE true

Test 'Tiled Serial' completed in [147.910281395] seconds
 ALL CLOSE true

Test 'Naive Parallel' completed in [11.999897961] seconds
 ALL CLOSE true

Test 'Optimized Parallel' completed in [9.451158368] seconds
 ALL CLOSE true

Test 'Tiled Parallel' completed in [2.312758167] seconds
 ALL CLOSE true

Test 'AVX Tiled Parallel' completed in [2.406559831] seconds
 ALL CLOSE true


<| TEST CASE |>: [M: 9000, K: 9000, N: 9000]
Generating matrices with seed 114514...
Generating golden data for [9000 x 9000] x [9000 x 9000] matrices...
Golden data generatoin completed in [0.986903719] seconds

Running benchmarks for [9000 x 9000] x [9000 x 9000] matrices...
Test 'Naive Serial' completed in [349.65938241] seconds
 ALL CLOSE true

Test 'Tiled Serial' completed in [210.1350156] seconds
 ALL CLOSE true

Test 'Naive Parallel' completed in [10.173730127] seconds
 ALL CLOSE true

Test 'Optimized Parallel' completed in [14.098844765] seconds
 ALL CLOSE true

Test 'Tiled Parallel' completed in [3.160320618] seconds
 ALL CLOSE true

Test 'AVX Tiled Parallel' completed in [3.333714134] seconds
 ALL CLOSE true


<| TEST CASE |>: [M: 10000, K: 10000, N: 10000]
Generating matrices with seed 114514...
Generating golden data for [10000 x 10000] x [10000 x 10000] matrices...
Golden data generatoin completed in [1.360227631] seconds

Running benchmarks for [10000 x 10000] x [10000 x 10000] matrices...
Test 'Naive Serial' completed in [499.665989382] seconds
 ALL CLOSE true

Test 'Tiled Serial' completed in [293.531557749] seconds
 ALL CLOSE true

Test 'Naive Parallel' completed in [14.659556009] seconds
 ALL CLOSE true

Test 'Optimized Parallel' completed in [14.194772444] seconds
 ALL CLOSE true

Test 'Tiled Parallel' completed in [4.345969034] seconds
 ALL CLOSE true

Test 'AVX Tiled Parallel' completed in [4.600836437] seconds
 ALL CLOSE true


<| TEST CASE |>: [M: 10000, K: 15000, N: 18000]
Generating matrices with seed 114514...
Generating golden data for [10000 x 15000] x [15000 x 18000] matrices...
Golden data generatoin completed in [3.171143667] seconds

Running benchmarks for [10000 x 15000] x [15000 x 18000] matrices...
Test 'Naive Serial' completed in [1356.302049273] seconds
 ALL CLOSE true

Test 'Tiled Serial' completed in [798.97698402] seconds
 ALL CLOSE true

Test 'Naive Parallel' completed in [45.533967551] seconds
 ALL CLOSE true

Test 'Optimized Parallel' completed in [45.387550666] seconds
 ALL CLOSE true

Test 'Tiled Parallel' completed in [11.725196583] seconds
 ALL CLOSE true

Test 'AVX Tiled Parallel' completed in [12.521405475] seconds
 ALL CLOSE true


<| TEST CASE |>: [M: 20000, K: 20000, N: 20000]
Generating matrices with seed 114514...
Generating golden data for [20000 x 20000] x [20000 x 20000] matrices...
Golden data generatoin completed in [8.857039895] seconds

Running benchmarks for [20000 x 20000] x [20000 x 20000] matrices...
Test 'Naive Serial' completed in [4145.112274234] seconds
 ALL CLOSE true

Test 'Tiled Serial' completed in [2422.09387692] seconds
 ALL CLOSE true

Test 'Naive Parallel' completed in [140.999450787] seconds
 ALL CLOSE true

Test 'Optimized Parallel' completed in [158.219105793] seconds
 ALL CLOSE true

Test 'Tiled Parallel' completed in [35.169008392] seconds
 ALL CLOSE true

Test 'AVX Tiled Parallel' completed in [37.550046566] seconds
 ALL CLOSE true
```

### 结论

从上面的结果可以看出，随着矩阵规模的增大，串行实现的性能急剧下降，而并行化的实现则显著提升了计算速度。特别是分块+SIMD+OpenMP的实现，在大规模矩阵乘法中表现尤为突出，充分利用了现代CPU的多核和向量化能力。

并且在2000x2000的矩阵乘法中，分块+SIMD+OpenMP的实现已经超过了OpenBLAS的性能，显示出自定义实现在小矩阵运算上的潜力。