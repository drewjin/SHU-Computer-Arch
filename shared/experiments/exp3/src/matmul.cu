#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cstddef>
#include <cstdlib>

#include "custom_config.h"
#include "matmul.cuh"
#include "utils.h"

__global__ void naive_cuda_matmul_kernel(const double* A,
                                         const double* B, double* C,
                                         size_t M, size_t K,
                                         size_t N) {
  size_t row_a = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col_b = blockIdx.x * blockDim.x + threadIdx.x;

  if (row_a < M and col_b < N) {
    double acc = 0.0f;
    for (size_t col_a = 0; col_a < K; col_a++) {
      acc += A[row_a * K + col_a] * B[col_a * N + col_b];
    }
    C[row_a * N + col_b] = acc;
  }
}

Matrix naive_cuda_matmul(const Matrix& A, const Matrix& B, size_t M,
                         size_t K, size_t N) {
  double *cuA, *cuB, *cuC;

  CHECK_CUDA(cudaMalloc(&cuA, M * K * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&cuB, K * N * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&cuC, M * N * sizeof(double)));

  CHECK_CUDA(cudaMemcpy(cuA, A.data(), M * K * sizeof(double),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(cuB, B.data(), K * N * sizeof(double),
                        cudaMemcpyHostToDevice));

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // 核函数调用后立即检查启动错误
  naive_cuda_matmul_kernel<<<grid, block>>>(cuA, cuB, cuC, M, K, N);
  CHECK_CUDA(cudaGetLastError());  // 检查核函数启动错误 [[9]]

  // 同步设备并检查执行错误
  CHECK_CUDA(cudaDeviceSynchronize());  // 确保核函数执行完成 [[4]]

  Matrix C(M * N, 0.0f);
  CHECK_CUDA(cudaMemcpy(C.data(), cuC, M * N * sizeof(double),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(cuA));
  CHECK_CUDA(cudaFree(cuB));
  CHECK_CUDA(cudaFree(cuC));

  return C;
}

Matrix tiled_cuda_matmul(const Matrix& A, const Matrix& B, size_t M,
                         size_t K, size_t N) {
  return Matrix();
}

Matrix cublas_matmul(const Matrix& A, const Matrix& B, size_t M,
                     size_t K, size_t N) {
  return Matrix();
}

Matrix cutlass_matmul(const Matrix& A, const Matrix& B, size_t M,
                      size_t K, size_t N) {
  return Matrix();
}

Matrix cute_matmul(const Matrix& A, const Matrix& B, size_t M,
                   size_t K, size_t N) {
  return Matrix();
}
