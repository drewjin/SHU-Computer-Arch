#include <mpi.h>
#include <vector>
#include <string>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

// 假设 CountryData 是一个结构体，存储国家 GDP 数据
struct CountryData {
  std::string countryCode;
  double gdp[61];  // 1960-2020 共 61 年数据
};

// 模拟全局数据集
std::vector<CountryData> loadData() {
  std::vector<CountryData> data;
  // 假设有 10 个国家，每个国家有 61 年的 GDP 数据
  for (int i = 0; i < 10; ++i) {
    CountryData country;
    country.countryCode = "C" + std::to_string(i);
    for (int j = 0; j < 61; ++j) {
      country.gdp[j] = 1000 + (rand() % 5000);  // 生成 GDP 数据
    }
    data.push_back(country);
  }
  return data;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<CountryData> globalData;
  if (rank == 0) {
    // 只有 root 进程加载数据
    globalData = loadData();
  }

  // 广播数据集大小给所有进程
  int totalCountries = 0;
  if (rank == 0) {
    totalCountries = globalData.size();
  }
  MPI_Bcast(&totalCountries, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // 计算每个进程应该处理的数据量
  int localCount = totalCountries / size;
  int remaining = totalCountries % size;
  if (rank < remaining) localCount += 1;  // 处理剩余的国家数据

  std::vector<CountryData> localData(localCount);

  // 让 root 进程发送数据给其他进程
  int offset = 0;
  if (rank == 0) {
    for (int i = 1; i < size; ++i) {
      int count = totalCountries / size + (i < remaining ? 1 : 0);
      MPI_Send(&globalData[offset], count * sizeof(CountryData), MPI_BYTE, i, 0,
               MPI_COMM_WORLD);
      offset += count;
    }
    localData.assign(globalData.begin(), globalData.begin() + localCount);
  } else {
    MPI_Recv(localData.data(), localCount * sizeof(CountryData), MPI_BYTE, 0, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // 每个进程绘制自己负责的数据
  for (const auto& country : localData) {
    std::vector<double> years(61);
    std::vector<double> gdp(country.gdp, country.gdp + 61);
    for (int i = 0; i < 61; ++i) years[i] = 1960 + i;

    plt::plot(years, gdp);
    plt::title("GDP per capita - " + country.countryCode);

    std::string filename = "plots/Country_" + country.countryCode + "_rank" +
                           std::to_string(rank) + ".png";
    plt::save(filename);
    plt::close();
  }

  MPI_Finalize();
  return 0;
}
