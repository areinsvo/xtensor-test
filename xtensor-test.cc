#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include <time.h>

#include "MatriplXT.h"

// /usr/local/Cellar/gcc/7.1.0/bin/c++-7 -O2 -mavx -DXTENSOR_USE_XSIMD -I /Users/cerati/miniconda3/include/ xtensor-test.cc -o xtensor-test.exe

void test_v0(size_t NN, std::array<xt::xarray<float>, 36>& input, int type) {

  MatriplXT66 A;
  for (size_t i=0;i<36;++i) A[i] = input[i];

  MatriplXT66 B;
  for (size_t i=0;i<36;++i) B[i] = input[i];

  const clock_t begin = clock();
  MatriplXT66 C;
  if (type==1) MultiplyXT66Loop(A, B, C);
  else if (type==2) MultiplyXT66LoopTile(A, B, C);
  else MultiplyXT66(A, B, C);
  const clock_t end = clock();

  for (size_t nn=0; nn<1/*NN*/; nn++) {
    // std::cout << "nn=" << nn << std::endl;
    // std::cout << "A" << std::endl;
    // A.print(std::cout,nn);
    // std::cout << "B" << std::endl;
    // B.print(std::cout,nn);
    std::cout << "C" << std::endl;
    C.print(std::cout,nn);
  }

  float time = float(end-begin)/CLOCKS_PER_SEC;
  if (type==1) std::cout << "v0 -- time for NN=" << NN << " multiplications is " << time << " s (loop version), i.e. per track [s]=" << time/float(NN) << std::endl;
  else if (type==2) std::cout << "v0 -- time for NN=" << NN << " multiplications is " << time << " s (loop-tile version), i.e. per track [s]=" << time/float(NN) << std::endl;
  else std::cout << "v0 -- time for NN=" << NN << " multiplications is " << time << " s, i.e. per track [s]=" << time/float(NN) << std::endl;
}

void test_v1(size_t NN, std::array<xt::xarray<float>, 36>& input, int type) {

  MatriplXT66_v1 A(NN);
  for (size_t i=0;i<36;++i) A[i] = input[i];

  MatriplXT66_v1 B(NN);
  for (size_t i=0;i<36;++i) B[i] = input[i];

  const clock_t begin = clock();
  MatriplXT66_v1 C(NN);
  if (type==1) MultiplyXT66Loop(A, B, C);
  else if (type==2) MultiplyXT66Stack(A, B, C);
  else MultiplyXT66(A, B, C);
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

  float time = float(end-begin)/CLOCKS_PER_SEC;
  if (type==1) std::cout << "v1 -- time for NN=" << NN << " multiplications is " << time << " s (loop version), i.e. per track [s]=" << time/float(NN) << std::endl;
  else if (type==2) std::cout << "v1 -- time for NN=" << NN << " multiplications is " << time << " s (stack version), i.e. per track [s]=" << time/float(NN) << std::endl;
  else std::cout << "v1 -- time for NN=" << NN << " multiplications is " << time << " s, i.e. per track [s]=" << time/float(NN) << std::endl;
}

void test_v2(size_t NN, std::array<xt::xarray<float>, 36>& input, int type) {

  MatriplXT66_v2 A(NN);
  for (size_t i=0;i<36;++i) A[i] = input[i];

  MatriplXT66_v2 B(NN);
  for (size_t i=0;i<36;++i) B[i] = input[i];

  const clock_t begin = clock();
  MatriplXT66_v2 C(NN);
  if (type==1) MultiplyXT66Loop(A, B, C);
  else if (type==2) MultiplyXT66Stack(A, B, C);
  else MultiplyXT66(A, B, C);
  const clock_t end = clock();

  for (size_t nn=0; nn<1/*NN*/; nn++) {
    // std::cout << "nn=" << nn << std::endl;
    // std::cout << "A" << std::endl;
    // A.print(std::cout,nn);
    // std::cout << "B" << std::endl;
    // B.print(std::cout,nn);
    std::cout << "C" << std::endl;
    C.print(std::cout,nn);
  }

  float time = float(end-begin)/CLOCKS_PER_SEC;
  if (type==1) std::cout << "v2 -- time for NN=" << NN << " multiplications is " << time << " s (loop version), i.e. per track [s]=" << time/float(NN) << std::endl;
  else if (type==2) std::cout << "v2 -- time for NN=" << NN << " multiplications is " << time << " s (stack version), i.e. per track [s]=" << time/float(NN) << std::endl;
  else std::cout << "v2 -- time for NN=" << NN << " multiplications is " << time << " s, i.e. per track [s]=" << time/float(NN) << std::endl;
}

int main(int argc, char* argv[])
{
  constexpr size_t NN = 10000000;

  std::array<xt::xarray<float>, 36> input;
  for (size_t i=0;i<36;++i) input[i] = xt::random::randn<float>({NN});
  //for (size_t i=0;i<36;++i) input[i] = xt::linspace<float>(1.,100.,NN);fixme

  std::cout << "done preparing input" << std::endl;

  test_v0(NN,input,0);
  test_v0(NN,input,0);// do it twice to let it warm up...
  test_v0(NN,input,1);
  test_v0(NN,input,2);
  std::cout << std::endl;
  test_v1(NN,input,0);
  test_v1(NN,input,1);
  test_v1(NN,input,2);
  std::cout << std::endl;
  test_v2(NN,input,0);
  test_v2(NN,input,1);
  test_v2(NN,input,2);

  return 0;
}
