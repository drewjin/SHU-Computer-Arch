#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <stdexcept>

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

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
            (M + TILE_SIZE - 1) / TILE_SIZE);

  naive_cuda_matmul_kernel<<<grid, block>>>(cuA, cuB, cuC, M, K, N);
  CHECK_CUDA(cudaGetLastError());  

  CHECK_CUDA(cudaDeviceSynchronize());  

  Matrix C(M * N, 0.0f);
  CHECK_CUDA(cudaMemcpy(C.data(), cuC, M * N * sizeof(double),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(cuA));
  CHECK_CUDA(cudaFree(cuB));
  CHECK_CUDA(cudaFree(cuC));

  return C;
}

__global__ void tiled_cuda_matmul_kernel(const double* A,
                                        const double* B, double* C,
                                        size_t M, size_t K,
                                        size_t N) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  __shared__ double As[TILE_SIZE][TILE_SIZE];
  __shared__ double Bs[TILE_SIZE][TILE_SIZE];

  double sum = 0.0;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    if (row < M && (t * TILE_SIZE + tx) < K) {
      As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
    } else {
      As[ty][tx] = 0.0;
    }

    if ((t * TILE_SIZE + ty) < K && col < N) {
      Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
    } else {
      Bs[ty][tx] = 0.0;
    }

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

Matrix tiled_cuda_matmul(const Matrix& A, const Matrix& B, size_t M,
                         size_t K, size_t N) {
  double *cuA, *cuB, *cuC;

  CHECK_CUDA(cudaMalloc(&cuA, M * K * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&cuB, K * N * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&cuC, M * N * sizeof(double)));

  CHECK_CUDA(cudaMemcpy(cuA, A.data(), M * K * sizeof(double),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(cuB, B.data(), K * N * sizeof(double),
                        cudaMemcpyHostToDevice));

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
            (M + TILE_SIZE - 1) / TILE_SIZE);

  tiled_cuda_matmul_kernel<<<grid, block>>>(cuA, cuB, cuC, M, K, N);
  CHECK_CUDA(cudaGetLastError());  

  CHECK_CUDA(cudaDeviceSynchronize());  

  Matrix C(M * N, 0.0f);
  CHECK_CUDA(cudaMemcpy(C.data(), cuC, M * N * sizeof(double),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(cuA));
  CHECK_CUDA(cudaFree(cuB));
  CHECK_CUDA(cudaFree(cuC));

  return C;
}

Matrix cublas_matmul(const Matrix& A, const Matrix& B, size_t M,
                     size_t K, size_t N) {
  cublasHandle_t handle;
  cudaError_t    cudaStatus;
  cublasStatus_t cublasStatus;

  cublasStatus = cublasCreate(&handle);
  if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create cuBLAS handle");
  }

  float *d_A, *d_B, *d_C;
  cudaStatus = cudaMalloc(&d_A, M * K * sizeof(float));
  cudaStatus = cudaMalloc(&d_B, K * N * sizeof(float));
  cudaStatus = cudaMalloc(&d_C, M * N * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    cublasDestroy(handle);
    throw std::runtime_error("Failed to allocate device memory");
  }

  cublasStatus =
    cublasSetMatrix(M, K, sizeof(float), A.data(), M, d_A, M);
  cublasStatus =
    cublasSetMatrix(K, N, sizeof(float), B.data(), K, d_B, K);
  if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    throw std::runtime_error("Failed to copy data to device");
  }

  float  alpha = 1.0f;
  float  beta  = 0.0f;
  float* A_ptr = d_A;  
  float* B_ptr = d_B;  
  float* C_ptr = d_C;  

  cublasStatus =
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                B_ptr, N,  
                A_ptr, K, &beta, C_ptr, N);
  if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    throw std::runtime_error("cuBLAS kernel execution failed");
  }

  Matrix C(M, N);
  cublasStatus =
    cublasGetMatrix(M, N, sizeof(float), d_C, M, C.data(), M);
  if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    throw std::runtime_error("Failed to copy result to host");
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(handle);

  return C;
}

Matrix cutlass_matmul(const Matrix& A, const Matrix& B, size_t M,
                      size_t K, size_t N) {
  return Matrix();
}

Matrix cute_matmul(const Matrix& A, const Matrix& B, size_t M,
                   size_t K, size_t N) {
  return Matrix();
}
