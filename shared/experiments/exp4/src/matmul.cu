#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

Matrix cublas_matmul(const Matrix& A, const Matrix& B,
           size_t M, size_t K, size_t N) {
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(double)));

  // Copy matrices from host to device
  CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(double), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B.data(), K * N * sizeof(double), cudaMemcpyHostToDevice));

  // Scalar factors
  const double alpha = 1.0;
  const double beta = 0.0;

  // Perform matrix multiplication: C = alpha*A*B + beta*C
  // Note: cuBLAS uses column-major order, but our matrices are row-major
  // So we compute B*A instead of A*B (effectively transposing the operation)
  CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
               N, M, K,
               &alpha,
               d_B, N,
               d_A, K,
               &beta,
               d_C, N));

  // Create result matrix and copy from device
  Matrix C(M * N, 0.0);
  CHECK_CUDA(cudaMemcpy(C.data(), d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));

  // Clean up
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  CHECK_CUBLAS(cublasDestroy(handle));

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
