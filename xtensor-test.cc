#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include <time.h>

#include "MatriplXT.h"

// /usr/local/Cellar/gcc/7.1.0/bin/c++-7 -O2 -mavx -DXTENSOR_USE_XSIMD -I /Users/cerati/miniconda3/include/ xtensor-test.cc -o xtensor-test.exe

constexpr size_t NN = 16*600000;//10000000;

void test_v0(std::array<xt::xarray<float>, 36>& input, int type) {

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

void test_v1(std::array<xt::xarray<float>, 36>& input, int type) {

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
    // std::cout << "nn=" << nn << std::endl;
    // std::cout << "A" << std::endl;
    // A.print(std::cout,nn);
    // std::cout << "B" << std::endl;
    // B.print(std::cout,nn);
    std::cout << "C" << std::endl;
    C.print(std::cout,nn);
  }

  float time = float(end-begin)/CLOCKS_PER_SEC;
  if (type==1) std::cout << "v1 -- time for NN=" << NN << " multiplications is " << time << " s (loop version), i.e. per track [s]=" << time/float(NN) << std::endl;
  else if (type==2) std::cout << "v1 -- time for NN=" << NN << " multiplications is " << time << " s (stack version), i.e. per track [s]=" << time/float(NN) << std::endl;
  else std::cout << "v1 -- time for NN=" << NN << " multiplications is " << time << " s, i.e. per track [s]=" << time/float(NN) << std::endl;
}

void test_v2(std::array<xt::xarray<float>, 36>& input, int type) {

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

void test_plain(std::array<xt::xarray<float>, 36>& input) {
  #ifdef USE_CALI
  CALI_CXX_MARK_FUNCTION;
  #endif
  float* Ax = new float[NN*36];
  float* Bx = new float[NN*36];
  float* Cx = new float[NN*36];
  for (size_t j=0;j<NN*36;++j) Cx[j]=0.;
  //
  for (size_t x=0;x<NN/16;++x) {
    for (size_t i=0;i<36;++i) {
      for (size_t n=0;n<16;++n) {
        Ax[n + i*16 + 16*36*x] = input[i](n+16*x);
        Bx[n + i*16 + 16*36*x] = input[i](n+16*x);
      }
    }
  }

  const clock_t begin = clock();
  for (size_t x = 0; x < NN/16; ++x) {
    const size_t Nx = x*16*36;
    for (size_t n = 0; n < 16; ++n) {
      Cx[Nx+16* 0+n] = Ax[Nx+16* 0+n]*Bx[Nx+16* 0+n] + Ax[Nx+16* 1+n]*Bx[Nx+16* 6+n] + Ax[Nx+16* 2+n]*Bx[Nx+16*12+n] + Ax[Nx+16* 3+n]*Bx[Nx+16*18+n] + Ax[Nx+16* 4+n]*Bx[Nx+16*24+n] + Ax[Nx+16* 5+n]*Bx[Nx+16*30+n];
      Cx[Nx+16* 1+n] = Ax[Nx+16* 0+n]*Bx[Nx+16* 1+n] + Ax[Nx+16* 1+n]*Bx[Nx+16* 7+n] + Ax[Nx+16* 2+n]*Bx[Nx+16*13+n] + Ax[Nx+16* 3+n]*Bx[Nx+16*19+n] + Ax[Nx+16* 4+n]*Bx[Nx+16*25+n] + Ax[Nx+16* 5+n]*Bx[Nx+16*31+n];
      Cx[Nx+16* 2+n] = Ax[Nx+16* 0+n]*Bx[Nx+16* 2+n] + Ax[Nx+16* 1+n]*Bx[Nx+16* 8+n] + Ax[Nx+16* 2+n]*Bx[Nx+16*14+n] + Ax[Nx+16* 3+n]*Bx[Nx+16*20+n] + Ax[Nx+16* 4+n]*Bx[Nx+16*26+n] + Ax[Nx+16* 5+n]*Bx[Nx+16*32+n];
      Cx[Nx+16* 3+n] = Ax[Nx+16* 0+n]*Bx[Nx+16* 3+n] + Ax[Nx+16* 1+n]*Bx[Nx+16* 9+n] + Ax[Nx+16* 2+n]*Bx[Nx+16*15+n] + Ax[Nx+16* 3+n]*Bx[Nx+16*21+n] + Ax[Nx+16* 4+n]*Bx[Nx+16*27+n] + Ax[Nx+16* 5+n]*Bx[Nx+16*33+n];
      Cx[Nx+16* 4+n] = Ax[Nx+16* 0+n]*Bx[Nx+16* 4+n] + Ax[Nx+16* 1+n]*Bx[Nx+16*10+n] + Ax[Nx+16* 2+n]*Bx[Nx+16*16+n] + Ax[Nx+16* 3+n]*Bx[Nx+16*22+n] + Ax[Nx+16* 4+n]*Bx[Nx+16*28+n] + Ax[Nx+16* 5+n]*Bx[Nx+16*34+n];
      Cx[Nx+16* 5+n] = Ax[Nx+16* 0+n]*Bx[Nx+16* 5+n] + Ax[Nx+16* 1+n]*Bx[Nx+16*11+n] + Ax[Nx+16* 2+n]*Bx[Nx+16*17+n] + Ax[Nx+16* 3+n]*Bx[Nx+16*23+n] + Ax[Nx+16* 4+n]*Bx[Nx+16*29+n] + Ax[Nx+16* 5+n]*Bx[Nx+16*35+n];
      Cx[Nx+16* 6+n] = Ax[Nx+16* 6+n]*Bx[Nx+16* 0+n] + Ax[Nx+16* 7+n]*Bx[Nx+16* 6+n] + Ax[Nx+16* 8+n]*Bx[Nx+16*12+n] + Ax[Nx+16* 9+n]*Bx[Nx+16*18+n] + Ax[Nx+16*10+n]*Bx[Nx+16*24+n] + Ax[Nx+16*11+n]*Bx[Nx+16*30+n];
      Cx[Nx+16* 7+n] = Ax[Nx+16* 6+n]*Bx[Nx+16* 1+n] + Ax[Nx+16* 7+n]*Bx[Nx+16* 7+n] + Ax[Nx+16* 8+n]*Bx[Nx+16*13+n] + Ax[Nx+16* 9+n]*Bx[Nx+16*19+n] + Ax[Nx+16*10+n]*Bx[Nx+16*25+n] + Ax[Nx+16*11+n]*Bx[Nx+16*31+n];
      Cx[Nx+16* 8+n] = Ax[Nx+16* 6+n]*Bx[Nx+16* 2+n] + Ax[Nx+16* 7+n]*Bx[Nx+16* 8+n] + Ax[Nx+16* 8+n]*Bx[Nx+16*14+n] + Ax[Nx+16* 9+n]*Bx[Nx+16*20+n] + Ax[Nx+16*10+n]*Bx[Nx+16*26+n] + Ax[Nx+16*11+n]*Bx[Nx+16*32+n];
      Cx[Nx+16* 9+n] = Ax[Nx+16* 6+n]*Bx[Nx+16* 3+n] + Ax[Nx+16* 7+n]*Bx[Nx+16* 9+n] + Ax[Nx+16* 8+n]*Bx[Nx+16*15+n] + Ax[Nx+16* 9+n]*Bx[Nx+16*21+n] + Ax[Nx+16*10+n]*Bx[Nx+16*27+n] + Ax[Nx+16*11+n]*Bx[Nx+16*33+n];
      Cx[Nx+16*10+n] = Ax[Nx+16* 6+n]*Bx[Nx+16* 4+n] + Ax[Nx+16* 7+n]*Bx[Nx+16*10+n] + Ax[Nx+16* 8+n]*Bx[Nx+16*16+n] + Ax[Nx+16* 9+n]*Bx[Nx+16*22+n] + Ax[Nx+16*10+n]*Bx[Nx+16*28+n] + Ax[Nx+16*11+n]*Bx[Nx+16*34+n];
      Cx[Nx+16*11+n] = Ax[Nx+16* 6+n]*Bx[Nx+16* 5+n] + Ax[Nx+16* 7+n]*Bx[Nx+16*11+n] + Ax[Nx+16* 8+n]*Bx[Nx+16*17+n] + Ax[Nx+16* 9+n]*Bx[Nx+16*23+n] + Ax[Nx+16*10+n]*Bx[Nx+16*29+n] + Ax[Nx+16*11+n]*Bx[Nx+16*35+n];
      Cx[Nx+16*12+n] = Ax[Nx+16*12+n]*Bx[Nx+16* 0+n] + Ax[Nx+16*13+n]*Bx[Nx+16* 6+n] + Ax[Nx+16*14+n]*Bx[Nx+16*12+n] + Ax[Nx+16*15+n]*Bx[Nx+16*18+n] + Ax[Nx+16*16+n]*Bx[Nx+16*24+n] + Ax[Nx+16*17+n]*Bx[Nx+16*30+n];
      Cx[Nx+16*13+n] = Ax[Nx+16*12+n]*Bx[Nx+16* 1+n] + Ax[Nx+16*13+n]*Bx[Nx+16* 7+n] + Ax[Nx+16*14+n]*Bx[Nx+16*13+n] + Ax[Nx+16*15+n]*Bx[Nx+16*19+n] + Ax[Nx+16*16+n]*Bx[Nx+16*25+n] + Ax[Nx+16*17+n]*Bx[Nx+16*31+n];
      Cx[Nx+16*14+n] = Ax[Nx+16*12+n]*Bx[Nx+16* 2+n] + Ax[Nx+16*13+n]*Bx[Nx+16* 8+n] + Ax[Nx+16*14+n]*Bx[Nx+16*14+n] + Ax[Nx+16*15+n]*Bx[Nx+16*20+n] + Ax[Nx+16*16+n]*Bx[Nx+16*26+n] + Ax[Nx+16*17+n]*Bx[Nx+16*32+n];
      Cx[Nx+16*15+n] = Ax[Nx+16*12+n]*Bx[Nx+16* 3+n] + Ax[Nx+16*13+n]*Bx[Nx+16* 9+n] + Ax[Nx+16*14+n]*Bx[Nx+16*15+n] + Ax[Nx+16*15+n]*Bx[Nx+16*21+n] + Ax[Nx+16*16+n]*Bx[Nx+16*27+n] + Ax[Nx+16*17+n]*Bx[Nx+16*33+n];
      Cx[Nx+16*16+n] = Ax[Nx+16*12+n]*Bx[Nx+16* 4+n] + Ax[Nx+16*13+n]*Bx[Nx+16*10+n] + Ax[Nx+16*14+n]*Bx[Nx+16*16+n] + Ax[Nx+16*15+n]*Bx[Nx+16*22+n] + Ax[Nx+16*16+n]*Bx[Nx+16*28+n] + Ax[Nx+16*17+n]*Bx[Nx+16*34+n];
      Cx[Nx+16*17+n] = Ax[Nx+16*12+n]*Bx[Nx+16* 5+n] + Ax[Nx+16*13+n]*Bx[Nx+16*11+n] + Ax[Nx+16*14+n]*Bx[Nx+16*17+n] + Ax[Nx+16*15+n]*Bx[Nx+16*23+n] + Ax[Nx+16*16+n]*Bx[Nx+16*29+n] + Ax[Nx+16*17+n]*Bx[Nx+16*35+n];
      Cx[Nx+16*18+n] = Ax[Nx+16*18+n]*Bx[Nx+16* 0+n] + Ax[Nx+16*19+n]*Bx[Nx+16* 6+n] + Ax[Nx+16*20+n]*Bx[Nx+16*12+n] + Ax[Nx+16*21+n]*Bx[Nx+16*18+n] + Ax[Nx+16*22+n]*Bx[Nx+16*24+n] + Ax[Nx+16*23+n]*Bx[Nx+16*30+n];
      Cx[Nx+16*19+n] = Ax[Nx+16*18+n]*Bx[Nx+16* 1+n] + Ax[Nx+16*19+n]*Bx[Nx+16* 7+n] + Ax[Nx+16*20+n]*Bx[Nx+16*13+n] + Ax[Nx+16*21+n]*Bx[Nx+16*19+n] + Ax[Nx+16*22+n]*Bx[Nx+16*25+n] + Ax[Nx+16*23+n]*Bx[Nx+16*31+n];
      Cx[Nx+16*20+n] = Ax[Nx+16*18+n]*Bx[Nx+16* 2+n] + Ax[Nx+16*19+n]*Bx[Nx+16* 8+n] + Ax[Nx+16*20+n]*Bx[Nx+16*14+n] + Ax[Nx+16*21+n]*Bx[Nx+16*20+n] + Ax[Nx+16*22+n]*Bx[Nx+16*26+n] + Ax[Nx+16*23+n]*Bx[Nx+16*32+n];
      Cx[Nx+16*21+n] = Ax[Nx+16*18+n]*Bx[Nx+16* 3+n] + Ax[Nx+16*19+n]*Bx[Nx+16* 9+n] + Ax[Nx+16*20+n]*Bx[Nx+16*15+n] + Ax[Nx+16*21+n]*Bx[Nx+16*21+n] + Ax[Nx+16*22+n]*Bx[Nx+16*27+n] + Ax[Nx+16*23+n]*Bx[Nx+16*33+n];
      Cx[Nx+16*22+n] = Ax[Nx+16*18+n]*Bx[Nx+16* 4+n] + Ax[Nx+16*19+n]*Bx[Nx+16*10+n] + Ax[Nx+16*20+n]*Bx[Nx+16*16+n] + Ax[Nx+16*21+n]*Bx[Nx+16*22+n] + Ax[Nx+16*22+n]*Bx[Nx+16*28+n] + Ax[Nx+16*23+n]*Bx[Nx+16*34+n];
      Cx[Nx+16*23+n] = Ax[Nx+16*18+n]*Bx[Nx+16* 5+n] + Ax[Nx+16*19+n]*Bx[Nx+16*11+n] + Ax[Nx+16*20+n]*Bx[Nx+16*17+n] + Ax[Nx+16*21+n]*Bx[Nx+16*23+n] + Ax[Nx+16*22+n]*Bx[Nx+16*29+n] + Ax[Nx+16*23+n]*Bx[Nx+16*35+n];
      Cx[Nx+16*24+n] = Ax[Nx+16*24+n]*Bx[Nx+16* 0+n] + Ax[Nx+16*25+n]*Bx[Nx+16* 6+n] + Ax[Nx+16*26+n]*Bx[Nx+16*12+n] + Ax[Nx+16*27+n]*Bx[Nx+16*18+n] + Ax[Nx+16*28+n]*Bx[Nx+16*24+n] + Ax[Nx+16*29+n]*Bx[Nx+16*30+n];
      Cx[Nx+16*25+n] = Ax[Nx+16*24+n]*Bx[Nx+16* 1+n] + Ax[Nx+16*25+n]*Bx[Nx+16* 7+n] + Ax[Nx+16*26+n]*Bx[Nx+16*13+n] + Ax[Nx+16*27+n]*Bx[Nx+16*19+n] + Ax[Nx+16*28+n]*Bx[Nx+16*25+n] + Ax[Nx+16*29+n]*Bx[Nx+16*31+n];
      Cx[Nx+16*26+n] = Ax[Nx+16*24+n]*Bx[Nx+16* 2+n] + Ax[Nx+16*25+n]*Bx[Nx+16* 8+n] + Ax[Nx+16*26+n]*Bx[Nx+16*14+n] + Ax[Nx+16*27+n]*Bx[Nx+16*20+n] + Ax[Nx+16*28+n]*Bx[Nx+16*26+n] + Ax[Nx+16*29+n]*Bx[Nx+16*32+n];
      Cx[Nx+16*27+n] = Ax[Nx+16*24+n]*Bx[Nx+16* 3+n] + Ax[Nx+16*25+n]*Bx[Nx+16* 9+n] + Ax[Nx+16*26+n]*Bx[Nx+16*15+n] + Ax[Nx+16*27+n]*Bx[Nx+16*21+n] + Ax[Nx+16*28+n]*Bx[Nx+16*27+n] + Ax[Nx+16*29+n]*Bx[Nx+16*33+n];
      Cx[Nx+16*28+n] = Ax[Nx+16*24+n]*Bx[Nx+16* 4+n] + Ax[Nx+16*25+n]*Bx[Nx+16*10+n] + Ax[Nx+16*26+n]*Bx[Nx+16*16+n] + Ax[Nx+16*27+n]*Bx[Nx+16*22+n] + Ax[Nx+16*28+n]*Bx[Nx+16*28+n] + Ax[Nx+16*29+n]*Bx[Nx+16*34+n];
      Cx[Nx+16*29+n] = Ax[Nx+16*24+n]*Bx[Nx+16* 5+n] + Ax[Nx+16*25+n]*Bx[Nx+16*11+n] + Ax[Nx+16*26+n]*Bx[Nx+16*17+n] + Ax[Nx+16*27+n]*Bx[Nx+16*23+n] + Ax[Nx+16*28+n]*Bx[Nx+16*29+n] + Ax[Nx+16*29+n]*Bx[Nx+16*35+n];
      Cx[Nx+16*30+n] = Ax[Nx+16*30+n]*Bx[Nx+16* 0+n] + Ax[Nx+16*31+n]*Bx[Nx+16* 6+n] + Ax[Nx+16*32+n]*Bx[Nx+16*12+n] + Ax[Nx+16*33+n]*Bx[Nx+16*18+n] + Ax[Nx+16*34+n]*Bx[Nx+16*24+n] + Ax[Nx+16*35+n]*Bx[Nx+16*30+n];
      Cx[Nx+16*31+n] = Ax[Nx+16*30+n]*Bx[Nx+16* 1+n] + Ax[Nx+16*31+n]*Bx[Nx+16* 7+n] + Ax[Nx+16*32+n]*Bx[Nx+16*13+n] + Ax[Nx+16*33+n]*Bx[Nx+16*19+n] + Ax[Nx+16*34+n]*Bx[Nx+16*25+n] + Ax[Nx+16*35+n]*Bx[Nx+16*31+n];
      Cx[Nx+16*32+n] = Ax[Nx+16*30+n]*Bx[Nx+16* 2+n] + Ax[Nx+16*31+n]*Bx[Nx+16* 8+n] + Ax[Nx+16*32+n]*Bx[Nx+16*14+n] + Ax[Nx+16*33+n]*Bx[Nx+16*20+n] + Ax[Nx+16*34+n]*Bx[Nx+16*26+n] + Ax[Nx+16*35+n]*Bx[Nx+16*32+n];
      Cx[Nx+16*33+n] = Ax[Nx+16*30+n]*Bx[Nx+16* 3+n] + Ax[Nx+16*31+n]*Bx[Nx+16* 9+n] + Ax[Nx+16*32+n]*Bx[Nx+16*15+n] + Ax[Nx+16*33+n]*Bx[Nx+16*21+n] + Ax[Nx+16*34+n]*Bx[Nx+16*27+n] + Ax[Nx+16*35+n]*Bx[Nx+16*33+n];
      Cx[Nx+16*34+n] = Ax[Nx+16*30+n]*Bx[Nx+16* 4+n] + Ax[Nx+16*31+n]*Bx[Nx+16*10+n] + Ax[Nx+16*32+n]*Bx[Nx+16*16+n] + Ax[Nx+16*33+n]*Bx[Nx+16*22+n] + Ax[Nx+16*34+n]*Bx[Nx+16*28+n] + Ax[Nx+16*35+n]*Bx[Nx+16*34+n];
      Cx[Nx+16*35+n] = Ax[Nx+16*30+n]*Bx[Nx+16* 5+n] + Ax[Nx+16*31+n]*Bx[Nx+16*11+n] + Ax[Nx+16*32+n]*Bx[Nx+16*17+n] + Ax[Nx+16*33+n]*Bx[Nx+16*23+n] + Ax[Nx+16*34+n]*Bx[Nx+16*29+n] + Ax[Nx+16*35+n]*Bx[Nx+16*35+n];
    }
  }
  // for (size_t x = 0; x < NN/16; ++x) {
  //   const size_t Nx = x*16*36;
  //   for (size_t i = 0; i < 6; ++i) {
  //     for (size_t j = 0; j < 6; ++j) {
  //       for (size_t k = 0; k < 6; ++k) {
  //         for (size_t n = 0; n < 16; ++n) {
  //           Cx[ Nx + (i*6 + j)*16 + n ] += Ax[ Nx + (i*6 + k)*16 + n ] * Bx[ Nx + (k*6 + j)*16 + n];
  //         }
  //       }
  //     }
  //   }
  // }
  const clock_t end = clock();

  // std::cout << "Ax=" << std::endl
  //        << Ax[16*(0*6+0)] << " " << Ax[16*(0*6+1)] << " " << Ax[16*(0*6+2)] << " " << Ax[16*(0*6+3)] << " " << Ax[16*(0*6+4)] << " " << Ax[16*(0*6+5)] << std::endl
  //        << Ax[16*(1*6+0)] << " " << Ax[16*(1*6+1)] << " " << Ax[16*(1*6+2)] << " " << Ax[16*(1*6+3)] << " " << Ax[16*(1*6+4)] << " " << Ax[16*(1*6+5)] << std::endl
  //        << Ax[16*(2*6+0)] << " " << Ax[16*(2*6+1)] << " " << Ax[16*(2*6+2)] << " " << Ax[16*(2*6+3)] << " " << Ax[16*(2*6+4)] << " " << Ax[16*(2*6+5)] << std::endl
  //        << Ax[16*(3*6+0)] << " " << Ax[16*(3*6+1)] << " " << Ax[16*(3*6+2)] << " " << Ax[16*(3*6+3)] << " " << Ax[16*(3*6+4)] << " " << Ax[16*(3*6+5)] << std::endl
  //        << Ax[16*(4*6+0)] << " " << Ax[16*(4*6+1)] << " " << Ax[16*(4*6+2)] << " " << Ax[16*(4*6+3)] << " " << Ax[16*(4*6+4)] << " " << Ax[16*(4*6+5)] << std::endl
  //        << Ax[16*(5*6+0)] << " " << Ax[16*(5*6+1)] << " " << Ax[16*(5*6+2)] << " " << Ax[16*(5*6+3)] << " " << Ax[16*(5*6+4)] << " " << Ax[16*(5*6+5)] << std::endl;
  // std::cout << "Bx=" << std::endl
  //        << Bx[16*(0*6+0)] << " " << Bx[16*(0*6+1)] << " " << Bx[16*(0*6+2)] << " " << Bx[16*(0*6+3)] << " " << Bx[16*(0*6+4)] << " " << Bx[16*(0*6+5)] << std::endl
  //        << Bx[16*(1*6+0)] << " " << Bx[16*(1*6+1)] << " " << Bx[16*(1*6+2)] << " " << Bx[16*(1*6+3)] << " " << Bx[16*(1*6+4)] << " " << Bx[16*(1*6+5)] << std::endl
  //        << Bx[16*(2*6+0)] << " " << Bx[16*(2*6+1)] << " " << Bx[16*(2*6+2)] << " " << Bx[16*(2*6+3)] << " " << Bx[16*(2*6+4)] << " " << Bx[16*(2*6+5)] << std::endl
  //        << Bx[16*(3*6+0)] << " " << Bx[16*(3*6+1)] << " " << Bx[16*(3*6+2)] << " " << Bx[16*(3*6+3)] << " " << Bx[16*(3*6+4)] << " " << Bx[16*(3*6+5)] << std::endl
  //        << Bx[16*(4*6+0)] << " " << Bx[16*(4*6+1)] << " " << Bx[16*(4*6+2)] << " " << Bx[16*(4*6+3)] << " " << Bx[16*(4*6+4)] << " " << Bx[16*(4*6+5)] << std::endl
  //        << Bx[16*(5*6+0)] << " " << Bx[16*(5*6+1)] << " " << Bx[16*(5*6+2)] << " " << Bx[16*(5*6+3)] << " " << Bx[16*(5*6+4)] << " " << Bx[16*(5*6+5)] << std::endl;
  std::cout << "Cx=" << std::endl
            << Cx[16*(0*6+0)] << " " << Cx[16*(0*6+1)] << " " << Cx[16*(0*6+2)] << " " << Cx[16*(0*6+3)] << " " << Cx[16*(0*6+4)] << " " << Cx[16*(0*6+5)] << std::endl
            << Cx[16*(1*6+0)] << " " << Cx[16*(1*6+1)] << " " << Cx[16*(1*6+2)] << " " << Cx[16*(1*6+3)] << " " << Cx[16*(1*6+4)] << " " << Cx[16*(1*6+5)] << std::endl
            << Cx[16*(2*6+0)] << " " << Cx[16*(2*6+1)] << " " << Cx[16*(2*6+2)] << " " << Cx[16*(2*6+3)] << " " << Cx[16*(2*6+4)] << " " << Cx[16*(2*6+5)] << std::endl
            << Cx[16*(3*6+0)] << " " << Cx[16*(3*6+1)] << " " << Cx[16*(3*6+2)] << " " << Cx[16*(3*6+3)] << " " << Cx[16*(3*6+4)] << " " << Cx[16*(3*6+5)] << std::endl
            << Cx[16*(4*6+0)] << " " << Cx[16*(4*6+1)] << " " << Cx[16*(4*6+2)] << " " << Cx[16*(4*6+3)] << " " << Cx[16*(4*6+4)] << " " << Cx[16*(4*6+5)] << std::endl
            << Cx[16*(5*6+0)] << " " << Cx[16*(5*6+1)] << " " << Cx[16*(5*6+2)] << " " << Cx[16*(5*6+3)] << " " << Cx[16*(5*6+4)] << " " << Cx[16*(5*6+5)] << std::endl;
  float time = float(end-begin)/CLOCKS_PER_SEC;
  std::cout << "plain -- time for NN=" << NN << " multiplications is " << time << " s, i.e. per track [s]=" << time/float(NN) << std::endl;
  delete Ax, Bx, Cx;
}

int main(int argc, char* argv[])
{

  std::array<xt::xarray<float>, 36> input;
  for (size_t i=0;i<36;++i) input[i] = xt::random::randn<float>({NN});
  //for (size_t i=0;i<36;++i) input[i] = xt::linspace<float>(1.,100.,NN);fixme

  std::cout << "done preparing input" << std::endl;

  test_v0(input,0);
  test_v0(input,0);// do it twice to let it warm up...
  test_v0(input,1);
  test_v0(input,2);
  std::cout << std::endl;
  test_v1(input,0);
  test_v1(input,1);
  // test_v1(input,2);
  std::cout << std::endl;
  test_v2(input,0);
  test_v2(input,1);
  // test_v2(input,2);
  std::cout << std::endl;

  test_plain(input);

  return 0;
}
