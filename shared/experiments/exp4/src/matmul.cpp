#include "custom_config.h"
#include "utils.h"
#include "matmul.h"

#include <cstddef>
#include <omp.h>
#include <openblas/cblas.h>
#include <immintrin.h>

Matrix openblas_matmul(const Matrix& A, const Matrix& B, size_t M, size_t K, size_t N) {
  Matrix C(M * N, 0.0f);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
              A.data(), K, B.data(), N, 0.0f, C.data(), N);
  return C;
}

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

Matrix forloop_interleave_serial_matmul(const Matrix& A,
                                        const Matrix& B, size_t M,
                                        size_t K, size_t N) {
  Matrix C(M * N, 0.0f); 
  for (size_t row_a = 0; row_a < M; row_a++) {
    for (size_t col_b = 0; col_b < N; col_b++) {
      for (size_t col_a = 0; col_a < K; col_a++) {
        C[row_a * N + col_b] +=
          A[row_a * K + col_a] * B[col_a * N + col_b];
      }
    }
  }
  return C;
}

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