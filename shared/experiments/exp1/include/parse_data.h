#ifndef PARSE_DATA
#define PARSE_DATA

#include <cstdio>
#include <vector>
#include <string>

#include "mpi.h"

const size_t NUM_YEARS = 61;
const size_t START_YEAR = 1960;

struct CountryData {
  char countryCode[256];
  char countryName[256];
  char region[256];
  char incomeGroup[256];
  double gdp[NUM_YEARS];

  void PrintCountryData();
};

MPI_Datatype CreateCountryMpiData();

// std::vector<std::string> SplitCsvLine(const std::string& line);

std::vector<CountryData> ReadMetaData(const std::string& filename);

void ReadGdpData(const std::string& filename, std::vector<CountryData>& countries);

#endif