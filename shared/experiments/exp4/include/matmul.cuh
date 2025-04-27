#ifndef MATMUL_CUH
#  define MATMUL_CUH

#  include "utils.h"
#  include <cstddef>

Matrix naive_cuda_matmul(const Matrix& A, const Matrix& B, size_t M,
                         size_t K, size_t N);

Matrix tiled_cuda_matmul(const Matrix& A, const Matrix& B, size_t M,
                         size_t K, size_t N);

Matrix cublas_matmul(const Matrix& A, const Matrix& B, size_t M,
                     size_t K, size_t N);

Matrix cutlass_matmul(const Matrix& A, const Matrix& B, size_t M,
                      size_t K, size_t N);

Matrix cute_matmul(const Matrix& A, const Matrix& B, size_t M,
                   size_t K, size_t N);

#endif