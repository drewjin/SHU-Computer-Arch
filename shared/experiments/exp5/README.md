# CUDA实验

## 要求

编写如下程序：随机生成两个大小为 10000 的数组，设为 A 数组和 B 数组，设结果数组 C，利用 CUDA 计算：

$$
C[i] =
\begin{cases}
A[i] + B[i], & i \text{是偶数} \\
A[i], & i \text{是奇数}
\end{cases},
0 < i < 10000
$$

要求在 CUDA 的 Kernel 函数中不使用 `if` 判断语句，结果数组 C 输出到文件中。分析比较“使用 GPU”和“不使用 GPU”这两种版本的程序的性能。
