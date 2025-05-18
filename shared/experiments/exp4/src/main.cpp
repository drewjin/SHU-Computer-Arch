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
      // auto* func_ptr = func.target<Matrix (*)(
      //   const Matrix&, const Matrix&, size_t, size_t, size_t)>();
      // if ((M > 3000 and K > 3000 and N > 3000) and
      //      (func_ptr and
      //       (*func_ptr == naive_serial_matmul or
      //        *func_ptr == tiled_serial_matmul or
      //        *func_ptr == forloop_interleave_serial_matmul))) {
      //   fmt::print("Overpass task, too large for serial matmul.\n");
      //   continue;
      // }
      auto result =
        run_benchmark(name, func, matA, matB, matGolden, M, K, N);
      fmt::print(
        "Test '{}' completed in [{}] seconds\n ALL CLOSE {}\n\n",
        result.test_name, result.execution_time, result.all_close);
    }
  }

  return 0;
}