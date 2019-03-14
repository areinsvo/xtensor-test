#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include <time.h>

#include "MatriplXT.h"

// /usr/local/Cellar/gcc/7.1.0/bin/c++-7 -O2 -mavx -DXTENSOR_USE_XSIMD -I /Users/cerati/miniconda3/include/ xtensor-test.cc -o xtensor-test.exe

int main(int argc, char* argv[])
{
  constexpr size_t NN = 10000000;

  MatriplXT66 A;
  for (size_t i=0;i<36;++i) A[i] = xt::random::randn<float>({NN});
  // for (size_t i=0;i<36;++i) A[i] = xt::ones<float>({NN});

  MatriplXT66 B;
  for (size_t i=0;i<36;++i) B[i] = xt::random::randn<float>({NN});
  // for (size_t i=0;i<36;++i) B[i] = xt::ones<float>({NN});

  const clock_t begin = clock();

  MatriplXT66 C;
  MultiplyXT66(A, B, C);

  const clock_t end = clock();

  for (size_t nn=0; nn<1/*NN*/; nn++) {
    std::cout << "nn=" << nn << std::endl;
    std::cout << "A" << std::endl;
    A.print(std::cout,nn);
    std::cout << "B" << std::endl;
    B.print(std::cout,nn);
    std::cout << "C" << std::endl;
    C.print(std::cout,nn);
  }

  std::cout << "time for NN=" << NN << " multiplications is " << float(end-begin)/CLOCKS_PER_SEC << " s" << std::endl;

  return 0;
}
