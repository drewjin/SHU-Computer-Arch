#include <cstddef>
#include <cstdio>
#include <cstring>
#include <format>
#include <unordered_map>
#include <mpi.h>
#include <utility>
#include <vector>
#include <string>
#include <cstdlib>
#include <iostream>

#include "csv.hpp"
#include "../include/parse_data.h"

void CountryData::PrintCountryData() {
  std::cout << std::format(
      "Country Code: {}\nName: {}\nRegion: {}\nIncome Group: {}\n",
      countryCode, countryName, region, incomeGroup)
      << "GDP: [";
  for (int i = 0; i < 61; i++) {
    std::cout << gdp[i];
  }
  std::cout << "]\n\n";
}

MPI_Datatype CreateCountryMpiData() {
  int blocklengths[5] = {256, 256, 256, 256, 61};
  MPI_Datatype types[5] = {MPI_CHAR, MPI_CHAR, MPI_CHAR, MPI_CHAR, MPI_DOUBLE};
  MPI_Aint displacements[5];

  CountryData dummy;
  MPI_Aint base_address;
  MPI_Get_address(&dummy, &base_address);
  MPI_Get_address(&dummy.countryCode, &displacements[0]);
  MPI_Get_address(&dummy.countryName, &displacements[1]);
  MPI_Get_address(&dummy.region, &displacements[2]);
  MPI_Get_address(&dummy.incomeGroup, &displacements[3]);
  MPI_Get_address(&dummy.gdp, &displacements[4]);

  for (int i = 0; i < 5; i++) {
    displacements[i] = displacements[i] - base_address;
  }

  MPI_Datatype mpiCountryType;
  MPI_Type_create_struct(5, blocklengths, displacements, types,
                         &mpiCountryType);
  MPI_Type_commit(&mpiCountryType);
  return mpiCountryType;
}

std::vector<CountryData> ReadMetaData(const std::string& filename) {
  std::vector<CountryData> countries;
  csv::CSVReader reader(filename);
  auto names = reader.get_col_names();
  for (auto row : reader) {
    CountryData tempCountry;
    // "Country Code","Region","IncomeGroup","SpecialNotes","TableName"
    strncpy(tempCountry.countryCode, row["Country Code"].get<>().c_str(), 255);
    strncpy(tempCountry.countryName, row["TableName"].get<>().c_str(), 255);
    strncpy(tempCountry.region, row["Region"].get<>().c_str(), 255);
    strncpy(tempCountry.incomeGroup, row["IncomeGroup"].get<>().c_str(), 255);
    countries.push_back(tempCountry);
  }
  return countries;
}

void ReadGdpData(const std::string& filename,
                 std::vector<CountryData>& countries) {
  std::unordered_map<std::string, CountryData*> gdpMap;
  for (auto& c : countries) {
    gdpMap[c.countryCode] = &c;
  }

  // Pre-compute the required year strings once
  std::vector<std::string> yearFields;
  for (size_t i = 0; i < NUM_YEARS; ++i) {
    yearFields.push_back(std::to_string(START_YEAR + i));
  }

  csv::CSVReader reader(filename);
  for (auto& row : reader) {  // Note: use reference to avoid copying
    auto name = row["Country Code"].get<std::string>();
    auto it = gdpMap.find(name);
    if (it != gdpMap.end()) {
      auto* country = it->second;
      for (size_t i = 0; i < NUM_YEARS; ++i) {
        const auto& yearStr = yearFields[i];
        auto data = row[yearStr].get<>();
        country->gdp[i] = data == "" ? 0.0 : std::stod(data);  // Assuming gdp is sized to NUM_YEARS
      }
    }
  }
}