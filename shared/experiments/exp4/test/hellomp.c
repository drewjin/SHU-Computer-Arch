#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

// #define NAIVE
#define MANUAL_UNROLL

#if defined (NAIVE)
void comput(float *A, float *B, float *C) {
  int x, y;
  for (y = 0; y < 4; y++) {
    for (x = 0; x < 4; x++) {
      C[4 * y + x] = A[4 * y + 0] * B[4 * 0 + x] + A[4 * y + 1] * B[4 * 1 + x] +
                     A[4 * y + 2] * B[4 * 2 + x] + A[4 * y + 3] * B[4 * 3 + x];
    }
  }
}
#elif defined (MANUAL_UNROLL)
void comput(float *A, float *B, float *C) {
  int x, y;
  for (y = 0; y < 4; ++y) {
    for (x = 0; x < 4; ++x) {
      C[4 * y + x] = A[4 * y + 0] * B[4 * 0 + x] + A[4 * y + 1] * B[4 * 1 + x] +
                     A[4 * y + 2] * B[4 * 2 + x] + A[4 * y + 3] * B[4 * 3 + x];
    }
  }
}
#endif

void comput_optimized(float *A, float *B, float *C) {
#pragma omp parallel for simd collapse(2)
  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < 4; x++) {
      float sum = 0;
#pragma GCC unroll 4
      for (int k = 0; k < 4; k++) {
        sum += A[4 * y + k] * B[4 * x + k]; // B已转置
      }
      C[4 * y + x] = sum;
    }
  }
}

// 使用 AVX256 优化的 4x4 矩阵乘法
void comput_avx(float *A, float *B, float *C) {
  // 加载矩阵 B 的列到 4 个 AVX 寄存器（转置后的 B）
  __m256 B0 = _mm256_loadu_ps(&B[0]);  // [B00, B10, B20, B30, 0,0,0,0]
  __m256 B1 = _mm256_loadu_ps(&B[4]);  // [B01, B11, B21, B31, 0,0,0,0]
  __m256 B2 = _mm256_loadu_ps(&B[8]);  // [B02, B12, B22, B32, 0,0,0,0]
  __m256 B3 = _mm256_loadu_ps(&B[12]); // [B03, B13, B23, B33, 0,0,0,0]

  // 处理矩阵 A 的每一行
#pragma GCC unroll 4
  for (int i = 0; i < 4; i++) {
    // 加载 A 的一行（广播到 AVX 寄存器的 4 个位置）
    __m256 A_row = _mm256_broadcast_ss(&A[4 * i]);

    // 计算 C[i][0], C[i][1], C[i][2], C[i][3]
    __m256 C_row = _mm256_mul_ps(A_row, B0);
    A_row = _mm256_broadcast_ss(&A[4 * i + 1]);
    C_row = _mm256_fmadd_ps(A_row, B1, C_row);
    A_row = _mm256_broadcast_ss(&A[4 * i + 2]);
    C_row = _mm256_fmadd_ps(A_row, B2, C_row);
    A_row = _mm256_broadcast_ss(&A[4 * i + 3]);
    C_row = _mm256_fmadd_ps(A_row, B3, C_row);

    // 存储结果（只取前 4 个 float）
    float tmp[8];
    _mm256_storeu_ps(tmp, C_row);
    C[4 * i] = tmp[0];
    C[4 * i + 1] = tmp[1];
    C[4 * i + 2] = tmp[2];
    C[4 * i + 3] = tmp[3];
  }
}

int main() {
  double duration;
  clock_t s, f;
  int x, y, n;
  float A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float B[] = {0.1f, 0.2f,  0.3f,  0.4f,  0.5f,  0.6f,  0.7f,  0.8f,
               0.9f, 0.10f, 0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f};
  float C[16];

  // 串行版本
  s = clock();
  for (n = 0; n < 1000000; n++)
    comput(A, B, C);
  f = clock();
  duration = (double)(f - s) / CLOCKS_PER_SEC;
  printf("Result(Serial):\n");
  for (y = 0; y < 4; y++) {
    for (x = 0; x < 4; x++)
      printf("%f,", C[y * 4 + x]);
    printf("\n");
  }
  printf("\n");
  printf("Serial :%f\n", duration);

  // 并行版本(2线程)
  s = clock();
#pragma omp parallel for num_threads(2)
  for (n = 0; n < 1000000; n++)
    comput(A, B, C);
  f = clock();
  duration = (double)(f - s) / CLOCKS_PER_SEC;
  printf("Parallel 2:%f\n", duration);

  // 并行版本(4线程)
  s = clock();
#pragma omp parallel for num_threads(4)
  for (n = 0; n < 1000000; n++)
    comput(A, B, C);
  f = clock();
  duration = (double)(f - s) / CLOCKS_PER_SEC;
  printf("Parallel 4:%f\n", duration);

  // 并行版本(8线程)
  s = clock();
#pragma omp parallel for num_threads(8)
  for (n = 0; n < 1000000; n++)
    comput(A, B, C);
  f = clock();
  duration = (double)(f - s) / CLOCKS_PER_SEC;
  printf("Parallel 8:%f\n", duration);

  // 并行版本(16线程)
  s = clock();
#pragma omp parallel for num_threads(16)
  for (n = 0; n < 1000000; n++)
    comput(A, B, C);
  f = clock();
  duration = (double)(f - s) / CLOCKS_PER_SEC;
  printf("Parallel 16:%f\n", duration);

  // AVX 优化版本
  s = clock();
  for (n = 0; n < 1000000; n++)
    comput_avx(A, B, C);
  f = clock();
  duration = (double)(f - s) / CLOCKS_PER_SEC;
  printf("AVX-256: %f sec\n", duration);

  // 转置B矩阵
  float B_transposed[16];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      B_transposed[4 * j + i] = B[4 * i + j];

  // 优化后的并行计算
  s = clock();
#pragma omp parallel for schedule(dynamic, 10000) num_threads(16)
  for (int n = 0; n < 1000000; n++) {
    comput_optimized(A, B_transposed, C);
  }
  f = clock();
  duration = (double)(f - s) / CLOCKS_PER_SEC;
  printf("Optimized Parallel: %f sec\n", duration);

  // 检查结果
  printf("\nResult:\n");
  for (y = 0; y < 4; y++) {
    for (x = 0; x < 4; x++)
      printf("%f,", C[y * 4 + x]);
    printf("\n");
  }

  return 0;
}