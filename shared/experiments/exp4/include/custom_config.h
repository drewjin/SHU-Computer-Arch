#pragma once

#include <vector>
#include "utils.h"

#define USE_SIMD
#define USE_UNROLL
#define BLOCK_SIZE 32

const std::vector<MatrixSize> SIZES {
  { 1000, 1000, 1000 }, { 2000, 2000, 2000 }, { 3000, 3000, 3000 },
    { 4000, 4000, 4000 }, { 5000, 5000, 5000 }, { 6000, 6000, 6000 },
    { 7000, 7000, 7000 }, { 8000, 8000, 8000 }, { 9000, 9000, 9000 },
    { 10000, 10000, 10000 }, { 10000, 15000,  18000  }, {20000, 20000, 20000}
};
