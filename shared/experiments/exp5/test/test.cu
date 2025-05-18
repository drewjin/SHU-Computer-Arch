#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define M 1024
#define N 512
#define K 1024

// Kernel 函数，负责计算结果矩阵的一个元素
__global__ void multiplyKernel(const int* dev_A, const int* dev_B, int* dev_C) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int k = 0; k < K; k++)
        sum += dev_A[Row * K + k] * dev_B[k * N + Col];

    dev_C[Row * N + Col] = sum;
}

// 检查 C[x,y] 计算是否正确
void check_matrix(int x, int y, int* A, int* B, int* C) {
    int sum = 0;
    for (int i = 0; i < K; i++)
        sum += A[x * K + i] * B[i * N + y];

    if (sum == C[x * N + y])
        printf("C[%d,%d] is right\n", x, y);
    else
        printf("C[%d,%d]=%d, sum=%d\n", x, y, C[x * N + y], sum);
}

// 使用 CUDA 进行矩阵乘法
cudaError_t multiplyWithCuda(int* C, const int* A, const int* B) {
    int* dev_A = 0;
    int* dev_B = 0;
    int* dev_C = 0;
    cudaError_t cudaStatus;
    
    // 声明并初始化dim3变量，移到函数开头，在所有goto语句之前
    dim3 dimGrid(16, 32);   // Grid 维度
    dim3 dimBlock(32, 32);  // Block 维度

    // 设置使用第 0 号 GPU
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Any CUDA-capable GPU installed?");
        goto Error;
    }

    // 分配显存
    cudaStatus = cudaMalloc((void**)&dev_A, M * K * sizeof(int));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
    cudaStatus = cudaMalloc((void**)&dev_B, K * N * sizeof(int));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
    cudaStatus = cudaMalloc((void**)&dev_C, M * N * sizeof(int));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

    // 数据拷贝到显存
    cudaStatus = cudaMemcpy(dev_A, A, M * K * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
    cudaStatus = cudaMemcpy(dev_B, B, K * N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

    // 启动 kernel
    multiplyKernel<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C);

    // 检查 kernel 是否启动失败
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multiplyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // 等待 kernel 执行完成
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d!\n", cudaStatus);
        goto Error;
    }

    // 将结果从显存复制回内存
    cudaStatus = cudaMemcpy(C, dev_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

Error:
    // 释放显存资源
    cudaFree(dev_C);
    cudaFree(dev_A);
    cudaFree(dev_B);
    return cudaStatus;
}

// 初始化矩阵
void init(int rowNum, int colNum, int* matrix) {
    srand((unsigned int)time(0));
    for (int i = 0; i < rowNum * colNum; i++)
        matrix[i] = rand() % 200;
}

// 主函数
int main() {
    int* A = (int*)malloc(M * K * sizeof(int));  // A[M][K]
    int* B = (int*)malloc(K * N * sizeof(int));  // B[K][N]
    int* C = (int*)malloc(M * N * sizeof(int));  // 结果矩阵 C[M][N]

    // 初始化矩阵 A 和 B
    init(M, K, A);
    init(K, N, B);

    clock_t start_clock, finish_clock;  // 用于计时
    double duration_seconds;

    start_clock = clock();

    // 调用 CUDA 矩阵乘法函数
    cudaError_t cudaStatus = multiplyWithCuda(C, A, B);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multiplyWithCuda failed!");
        return 1;
    }

    finish_clock = clock();
    duration_seconds = (double)(finish_clock - start_clock) / CLOCKS_PER_SEC;
    printf("CUDA use time(s): %f\n", duration_seconds);
    printf("================================================\n");

    // 重置 CUDA 设备
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    // 检查结果是否正确
    check_matrix(1023, 511, A, B, C);

    // 释放内存
    free(A);
    free(B);
    free(C);

    return 0;
}