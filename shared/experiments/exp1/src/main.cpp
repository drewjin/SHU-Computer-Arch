#include <climits>
#include <cstddef>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "mpi.h"
#include "matplotlibcpp.h"
#include "../include/parse_data.h"
#include "../include/utils.h"

namespace plt = matplotlibcpp;

int main(int argc, char* argv[]) {
  // For debug
  // {
  //   int i = 0;
  //   std::cout << "Waiting for debugger to detach\n";
  //   while (0 == i) sleep(5);
  // }

  // Init MPI
  int rank, numProcs;
  // MPI_Init(&argc, &argv);
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  // Init variables
  auto mpiCountryType = CreateCountryMpiData();
  auto start_time = MPI_Wtime();
  std::vector<CountryData> fullData;
  size_t dataSize;

  // Host load data
  if (rank == 0) {
    const std::string META_DATA_FILE = "data/meta-data.csv";
    const std::string GDP_DATA_FILE = "data/gdp-data";
    fullData = ReadMetaData(META_DATA_FILE);
    ReadGdpData(GDP_DATA_FILE, fullData);
    dataSize = fullData.size();
    std::cout << std::format("Loaded {} countries\n", fullData.size());
  }

  MPI_Bcast(&dataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (dataSize <= 0) {
    if (rank == 0) {
      std::cerr << "Error: No valid data loaded\n";
    }
    MPI_Finalize();
    return 1;
  }

  int chunkSize = dataSize / numProcs;
  int remainder = dataSize % numProcs;
  std::vector<int> counts(numProcs, chunkSize);
  std::vector<int> displs(numProcs, 0);

  for (int i = 0; i < remainder; ++i) {
    counts[i]++;
  }

  for (int i = 1; i < numProcs; ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }

  std::vector<CountryData> localData(counts[rank]);

  MPI_Scatterv(fullData.data(), counts.data(), displs.data(), mpiCountryType,
               localData.data(), counts[rank], mpiCountryType, 0,
               MPI_COMM_WORLD);

  if (rank == 0) {
    const std::string PLOT_DIR = "./plots";
    ClearDirectory(PLOT_DIR);
    CreateDirectory(PLOT_DIR);
    CreateDirectory(PLOT_DIR + "/countries");
    CreateDirectory(PLOT_DIR + "/region");
    CreateDirectory(PLOT_DIR + "/income");
  } 
  MPI_Barrier(MPI_COMM_WORLD);

  for (const auto& country : localData) {
    std::vector<double> years(61);
    std::vector<double> gdp(country.gdp, country.gdp + 61);
    for (int i = 0; i < 61; ++i) years[i] = 1960 + i;
    plt::plot(years, gdp);
    plt::title(std::string(country.countryCode) + " GDP per capita");
    plt::save("./plots/countries/Country_" + std::string(country.countryCode) + ".png");
    plt::close();
  }

  std::set<std::string> regions, incomeGroups;
  if (rank == 0) {
    for (const auto& c : fullData) {
      if (strlen(c.region))
        regions.insert(c.region);
      if (strlen(c.incomeGroup))
        incomeGroups.insert(c.incomeGroup);
    }
  }

  std::vector<char> regionBuffer, incomeBuffer;
  int regionBufferSize = 0, incomeBufferSize = 0;
  if (rank == 0) {
    regionBuffer = SerializeStringSet(regions);
    incomeBuffer = SerializeStringSet(incomeGroups);
    regionBufferSize = regionBuffer.size();
    incomeBufferSize = incomeBuffer.size();
  }

  MPI_Bcast(&regionBufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&incomeBufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  regionBuffer.resize(regionBufferSize);
  incomeBuffer.resize(incomeBufferSize);
  MPI_Bcast(regionBuffer.data(), regionBufferSize, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(incomeBuffer.data(), incomeBufferSize, MPI_CHAR, 0, MPI_COMM_WORLD);
  std::set<std::string> allRegions =
      DeserializeStringSet(regionBuffer.data(), regionBufferSize);
  std::set<std::string> allIncomeGroups =
      DeserializeStringSet(incomeBuffer.data(), incomeBufferSize);

  std::map<std::string, std::vector<double>> regionSum, incomeSum;
  std::map<std::string, std::vector<int>> regionCounts, incomeCounts;
  for (const auto& r : allRegions) {
    regionSum[r] = std::vector<double>(61, 0.0);
    regionCounts[r] = std::vector<int>(61, 0);
  }
  for (const auto& ig : allIncomeGroups) {
    incomeSum[ig] = std::vector<double>(61, 0.0);
    incomeCounts[ig] = std::vector<int>(61, 0);
  }

  for (const auto& country : localData) {
    std::string region = country.region;
    std::string income = country.incomeGroup;
    for (int i = 0; i < 61; ++i) {
      double gdp = country.gdp[i];
      if (gdp <= 1e-9) continue; 
      if (allRegions.count(region)) {
        regionSum[region][i] += gdp;
        regionCounts[region][i]++;
      }
      if (allIncomeGroups.count(income)) {
        incomeSum[income][i] += gdp;
        incomeCounts[income][i]++;
      }
    }
  }

  for (const auto& r : allRegions) {
    MPI_Reduce((rank == 0) ? MPI_IN_PLACE : regionSum[r].data(),
               regionSum[r].data(), 61, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce((rank == 0) ? MPI_IN_PLACE : regionCounts[r].data(),
               regionCounts[r].data(), 61, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
  for (const auto& ig : allIncomeGroups) {
    MPI_Reduce((rank == 0) ? MPI_IN_PLACE : incomeSum[ig].data(),
               incomeSum[ig].data(), 61, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce((rank == 0) ? MPI_IN_PLACE : incomeCounts[ig].data(),
               incomeCounts[ig].data(), 61, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    for (const auto& [region, sum] : regionSum) {
      const auto& counts = regionCounts[region];
      std::vector<double> avg(61);
      for (int i = 0; i < 61; ++i) {
        avg[i] = (counts[i] > 0) ? sum[i] / counts[i] : 0.0;
      }
      std::vector<double> years(61);
      for (int i = 0; i < 61; ++i) years[i] = 1960 + i;
      plt::plot(years, avg);
      plt::title("Region: " + region);
      plt::save("./plots/region/Region_" + region + ".png");
      plt::close();
    }

    for (const auto& [ig, sum] : incomeSum) {
      const auto& counts = incomeCounts[ig];
      std::vector<double> avg(61);
      for (int i = 0; i < 61; ++i) {
        avg[i] = (counts[i] > 0) ? sum[i] / counts[i] : 0.0;
      }
      std::vector<double> years(61);
      for (int i = 0; i < 61; ++i) years[i] = 1960 + i;
      plt::plot(years, avg);
      plt::title("Income Group: " + ig);
      plt::save("./plots/income/IncomeGroup_" + ig + ".png");
      plt::close();
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double end_time = MPI_Wtime();
  if (rank == 0) {
    double total_time = end_time - start_time;
    std::cout << "Total time: " << total_time << " seconds\n";
    std::cout << "Run with different number of processes to compare times.\n";
    std::cout << "Parallel Speedup = T1 / Tp\n";
    std::cout << "Parallel Efficiency = Speedup / p\n";
  }

  MPI_Type_free(&mpiCountryType);
  MPI_Finalize();
  return 0;
}
