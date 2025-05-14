#pragma once

#include <cstddef>
#include "utils.h"

Matrix openblas_matmul(const Matrix& A, const Matrix& B, size_t M,
                       size_t K, size_t N);

Matrix naive_serial_matmul(const Matrix& A, const Matrix& B, size_t M,
                           size_t K, size_t N);


// Matrix forloop_interleave_serial_matmul(const Matrix& A,
//                                         const Matrix& B, size_t M,
//                                         size_t K, size_t N);


Matrix naive_parallel_matmul(const Matrix& A, const Matrix& B,
                             size_t M, size_t K, size_t N);


Matrix optimized_parallel_matmul(const Matrix& A, const Matrix& B,
                                 size_t M, size_t K, size_t N);

Matrix tiled_serial_matmul(const Matrix& A, const Matrix& B, size_t M,
                           size_t K, size_t N);

Matrix tiled_parallel_matmul(const Matrix& A, const Matrix& B,
                             size_t M, size_t K, size_t N);

Matrix tiled_parallel_matmul_avx512(const Matrix& A, const Matrix& B,
                                    size_t M, size_t K, size_t N);