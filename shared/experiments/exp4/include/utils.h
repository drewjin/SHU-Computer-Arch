#pragma once

#include <cstddef>
#include <string>
#include <cstdint>
#include <vector>
#include <functional>

#define CHECK_CUDA(call)                                     \
  do {                                                       \
    cudaError_t err = call;                                  \
    if (err != cudaSuccess) {                                \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, \
              __LINE__, cudaGetErrorString(err));            \
      exit(EXIT_FAILURE);                                    \
    }                                                        \
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
                              