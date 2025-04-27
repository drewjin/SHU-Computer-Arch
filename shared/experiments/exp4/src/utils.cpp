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

size_t nearest_length(size_t n) {
  return ((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
}

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
