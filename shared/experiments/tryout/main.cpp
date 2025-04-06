#include <format>
#include <iostream>
#include <string>

#include "matplotlibcpp.h"
#include "mpi.h"

namespace plt = matplotlibcpp;

int main(int argc, char * argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);  // 修正：应该传递 argc 和 argv
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    plt::plot({ 1, 2, 3, 4 });

    // 使用 std::format 的正确方式
    std::cout << std::format(">>>> Matplotlib from process [{}] of size [{}]\n", rank, size);

    // 直接使用 std::format 创建文件名
    std::string filename = std::format("{}.jpg", rank);
    plt::save(filename);

    MPI_Finalize();

    return 0;
}
