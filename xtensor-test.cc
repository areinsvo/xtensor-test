#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include <time.h>

constexpr size_t NN = 16*600000;//10000000;

#include "MatriplXT.h"

// /usr/local/Cellar/gcc/7.1.0/bin/c++-7 -O2 -mavx -DXTENSOR_USE_XSIMD -I /Users/cerati/miniconda3/include/ xtensor-test.cc -o xtensor-test.exe

void test_v0(const std::array<xt::xarray<float>, 36>& input, const int type) {

  MatriplXT66 A;
  for (size_t i=0;i<36;++i) A[i] = input[i];

  MatriplXT66 B;
  for (size_t i=0;i<36;++i) B[i] = input[i]+0.5;

  MatriplXT66 C;
  const clock_t begin = clock();
  if (type==1) MultiplyXT66Loop(A, B, C);
  else if (type==2) MultiplyXT66LoopTile(A, B, C);
  else MultiplyXT66(A, B, C);
  const clock_t end = clock();

  for (size_t nn=0; nn<1/*NN*/; nn++) {
    // std::cout << "nn=" << nn << std::endl;
    // // std::cout << std::scientific;
    // std::cout << "A" << std::endl;
    // A.print(std::cout,nn);
    // std::cout << "B" << std::endl;
    // B.print(std::cout,nn);
    std::cout << "C" << std::endl;
    C.print(std::cout,nn);
    // std::cout << std::fixed;
  }

  float time = float(end-begin)/CLOCKS_PER_SEC;
  if (type==1) std::cout << "v0 -- time for NN=" << NN << " multiplications is " << time << " s (loop version), i.e. per track [s]=" << time/float(NN) << std::endl;
  else if (type==2) std::cout << "v0 -- time for NN=" << NN << " multiplications is " << time << " s (loop-tile version), i.e. per track [s]=" << time/float(NN) << std::endl;
  else std::cout << "v0 -- time for NN=" << NN << " multiplications is " << time << " s, i.e. per track [s]=" << time/float(NN) << std::endl;
}

void test_v1(const std::array<xt::xarray<float>, 36>& input, const int type) {

  MatriplXT66_v1 A(NN);
  for (size_t i=0;i<36;++i) A[i] = input[i];

  MatriplXT66_v1 B(NN);
  for (size_t i=0;i<36;++i) B[i] = input[i]+0.5;

  MatriplXT66_v1 C(NN);
  const clock_t begin = clock();
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

void test_v2(const std::array<xt::xarray<float>, 36>& input, const int type) {

  MatriplXT66_v2 A(NN);
  for (size_t i=0;i<36;++i) A[i] = input[i];

  MatriplXT66_v2 B(NN);
  for (size_t i=0;i<36;++i) B[i] = input[i]+0.5;

  MatriplXT66_v2 C(NN);
  const clock_t begin = clock();
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

void test_v3(const std::array<xt::xarray<float>, 36>& input, const int type) {

  MatriplXT66_v3 A;
  for (size_t i=0;i<36;++i) A[i] = input[i];

  MatriplXT66_v3 B;
  for (size_t i=0;i<36;++i) B[i] = input[i]+0.5;

  MatriplXT66_v3 C;
  const clock_t begin = clock();
  // if (type==1) MultiplyXT66Loop(A, B, C);
  // else if (type==2) MultiplyXT66Stack(A, B, C);
  // else
  MultiplyXT66(A, B, C);
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
  // if (type==1) std::cout << "v3 -- time for NN=" << NN << " multiplications is " << time << " s (loop version), i.e. per track [s]=" << time/float(NN) << std::endl;
  // else if (type==2) std::cout << "v3 -- time for NN=" << NN << " multiplications is " << time << " s (stack version), i.e. per track [s]=" << time/float(NN) << std::endl;
  // else
  std::cout << "v3 -- time for NN=" << NN << " multiplications is " << time << " s, i.e. per track [s]=" << time/float(NN) << std::endl;
}


void test_plainArray_matrix(const std::array<xt::xarray<float>, 36>& input, const int type) {
  //
  float* Ax = new float[NN*36];
  float* Bx = new float[NN*36];
  float* Cx = new float[NN*36];
  for (size_t j=0;j<NN*36;++j) Cx[j]=0.;
  //
  // store in matrix order (i.e. all elements of a given matrix are contiguous)
  for (size_t x=0;x<NN;++x) {
    for (size_t i=0;i<36;++i) {
      Ax[i + 36*x] = input[i](x);
      Bx[i + 36*x] = input[i](x)+0.5;
    }
  }
  const clock_t begin = clock();
  for (size_t x = 0; x < NN; ++x) {
    const size_t Nx = x*36;
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        for (size_t k = 0; k < 6; ++k) {
          Cx[ Nx + (i*6 + j) ] += Ax[ Nx + (i*6 + k) ] * Bx[ Nx + (k*6 + j) ];
        }
      }
    }
  }
  const clock_t end = clock();

  // std::cout << "Ax=" << std::endl
  //        << Ax[(0*6+0)] << "\t" << Ax[(0*6+1)] << "\t" << Ax[(0*6+2)] << "\t" << Ax[(0*6+3)] << "\t" << Ax[(0*6+4)] << "\t" << Ax[(0*6+5)] << std::endl
  //        << Ax[(1*6+0)] << "\t" << Ax[(1*6+1)] << "\t" << Ax[(1*6+2)] << "\t" << Ax[(1*6+3)] << "\t" << Ax[(1*6+4)] << "\t" << Ax[(1*6+5)] << std::endl
  //        << Ax[(2*6+0)] << "\t" << Ax[(2*6+1)] << "\t" << Ax[(2*6+2)] << "\t" << Ax[(2*6+3)] << "\t" << Ax[(2*6+4)] << "\t" << Ax[(2*6+5)] << std::endl
  //        << Ax[(3*6+0)] << "\t" << Ax[(3*6+1)] << "\t" << Ax[(3*6+2)] << "\t" << Ax[(3*6+3)] << "\t" << Ax[(3*6+4)] << "\t" << Ax[(3*6+5)] << std::endl
  //        << Ax[(4*6+0)] << "\t" << Ax[(4*6+1)] << "\t" << Ax[(4*6+2)] << "\t" << Ax[(4*6+3)] << "\t" << Ax[(4*6+4)] << "\t" << Ax[(4*6+5)] << std::endl
  //        << Ax[(5*6+0)] << "\t" << Ax[(5*6+1)] << "\t" << Ax[(5*6+2)] << "\t" << Ax[(5*6+3)] << "\t" << Ax[(5*6+4)] << "\t" << Ax[(5*6+5)] << std::endl;
  // std::cout << "Bx=" << std::endl
  //        << Bx[(0*6+0)] << "\t" << Bx[(0*6+1)] << "\t" << Bx[(0*6+2)] << "\t" << Bx[(0*6+3)] << "\t" << Bx[(0*6+4)] << "\t" << Bx[(0*6+5)] << std::endl
  //        << Bx[(1*6+0)] << "\t" << Bx[(1*6+1)] << "\t" << Bx[(1*6+2)] << "\t" << Bx[(1*6+3)] << "\t" << Bx[(1*6+4)] << "\t" << Bx[(1*6+5)] << std::endl
  //        << Bx[(2*6+0)] << "\t" << Bx[(2*6+1)] << "\t" << Bx[(2*6+2)] << "\t" << Bx[(2*6+3)] << "\t" << Bx[(2*6+4)] << "\t" << Bx[(2*6+5)] << std::endl
  //        << Bx[(3*6+0)] << "\t" << Bx[(3*6+1)] << "\t" << Bx[(3*6+2)] << "\t" << Bx[(3*6+3)] << "\t" << Bx[(3*6+4)] << "\t" << Bx[(3*6+5)] << std::endl
  //        << Bx[(4*6+0)] << "\t" << Bx[(4*6+1)] << "\t" << Bx[(4*6+2)] << "\t" << Bx[(4*6+3)] << "\t" << Bx[(4*6+4)] << "\t" << Bx[(4*6+5)] << std::endl
  //        << Bx[(5*6+0)] << "\t" << Bx[(5*6+1)] << "\t" << Bx[(5*6+2)] << "\t" << Bx[(5*6+3)] << "\t" << Bx[(5*6+4)] << "\t" << Bx[(5*6+5)] << std::endl;
  std::cout << "Cx=" << std::endl
            << Cx[(0*6+0)] << "\t" << Cx[(0*6+1)] << "\t" << Cx[(0*6+2)] << "\t" << Cx[(0*6+3)] << "\t" << Cx[(0*6+4)] << "\t" << Cx[(0*6+5)] << std::endl
            << Cx[(1*6+0)] << "\t" << Cx[(1*6+1)] << "\t" << Cx[(1*6+2)] << "\t" << Cx[(1*6+3)] << "\t" << Cx[(1*6+4)] << "\t" << Cx[(1*6+5)] << std::endl
            << Cx[(2*6+0)] << "\t" << Cx[(2*6+1)] << "\t" << Cx[(2*6+2)] << "\t" << Cx[(2*6+3)] << "\t" << Cx[(2*6+4)] << "\t" << Cx[(2*6+5)] << std::endl
            << Cx[(3*6+0)] << "\t" << Cx[(3*6+1)] << "\t" << Cx[(3*6+2)] << "\t" << Cx[(3*6+3)] << "\t" << Cx[(3*6+4)] << "\t" << Cx[(3*6+5)] << std::endl
            << Cx[(4*6+0)] << "\t" << Cx[(4*6+1)] << "\t" << Cx[(4*6+2)] << "\t" << Cx[(4*6+3)] << "\t" << Cx[(4*6+4)] << "\t" << Cx[(4*6+5)] << std::endl
            << Cx[(5*6+0)] << "\t" << Cx[(5*6+1)] << "\t" << Cx[(5*6+2)] << "\t" << Cx[(5*6+3)] << "\t" << Cx[(5*6+4)] << "\t" << Cx[(5*6+5)] << std::endl;
  float time = float(end-begin)/CLOCKS_PER_SEC;
  std::cout << "plainArray_matrix -- time for NN=" << NN << " multiplications is " << time << " s, i.e. per track [s]=" << time/float(NN) << std::endl;

  delete Ax, Bx, Cx;
}

void test_plainArray_element(const std::array<xt::xarray<float>, 36>& input, const int type) {
  //
  float* Ax = new float[NN*36];
  float* Bx = new float[NN*36];
  float* Cx = new float[NN*36];
  for (size_t j=0;j<NN*36;++j) Cx[j]=0.;
  //
  // store in element order (i.e. all matrices for a given element are contiguous)
  for (size_t x=0;x<NN;++x) {
    for (size_t i=0;i<36;++i) {
      Ax[i*NN + x] = input[i](x);
      Bx[i*NN + x] = input[i](x)+0.5;
    }
  }
  const clock_t begin = clock();
  for (size_t x = 0; x < NN; ++x) {
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        for (size_t k = 0; k < 6; ++k) {
          Cx[ x + (i*6 + j)*NN ] += Ax[ x + (i*6 + k)*NN ] * Bx[ x + (k*6 + j)*NN ];
        }
      }
    }
  }
  const clock_t end = clock();

  // std::cout << "Ax=" << std::endl
  //        << Ax[NN*(0*6+0)] << "\t" << Ax[NN*(0*6+1)] << "\t" << Ax[NN*(0*6+2)] << "\t" << Ax[NN*(0*6+3)] << "\t" << Ax[NN*(0*6+4)] << "\t" << Ax[NN*(0*6+5)] << std::endl
  //        << Ax[NN*(1*6+0)] << "\t" << Ax[NN*(1*6+1)] << "\t" << Ax[NN*(1*6+2)] << "\t" << Ax[NN*(1*6+3)] << "\t" << Ax[NN*(1*6+4)] << "\t" << Ax[NN*(1*6+5)] << std::endl
  //        << Ax[NN*(2*6+0)] << "\t" << Ax[NN*(2*6+1)] << "\t" << Ax[NN*(2*6+2)] << "\t" << Ax[NN*(2*6+3)] << "\t" << Ax[NN*(2*6+4)] << "\t" << Ax[NN*(2*6+5)] << std::endl
  //        << Ax[NN*(3*6+0)] << "\t" << Ax[NN*(3*6+1)] << "\t" << Ax[NN*(3*6+2)] << "\t" << Ax[NN*(3*6+3)] << "\t" << Ax[NN*(3*6+4)] << "\t" << Ax[NN*(3*6+5)] << std::endl
  //        << Ax[NN*(4*6+0)] << "\t" << Ax[NN*(4*6+1)] << "\t" << Ax[NN*(4*6+2)] << "\t" << Ax[NN*(4*6+3)] << "\t" << Ax[NN*(4*6+4)] << "\t" << Ax[NN*(4*6+5)] << std::endl
  //        << Ax[NN*(5*6+0)] << "\t" << Ax[NN*(5*6+1)] << "\t" << Ax[NN*(5*6+2)] << "\t" << Ax[NN*(5*6+3)] << "\t" << Ax[NN*(5*6+4)] << "\t" << Ax[NN*(5*6+5)] << std::endl;
  // std::cout << "Bx=" << std::endl
  //        << Bx[NN*(0*6+0)] << "\t" << Bx[NN*(0*6+1)] << "\t" << Bx[NN*(0*6+2)] << "\t" << Bx[NN*(0*6+3)] << "\t" << Bx[NN*(0*6+4)] << "\t" << Bx[NN*(0*6+5)] << std::endl
  //        << Bx[NN*(1*6+0)] << "\t" << Bx[NN*(1*6+1)] << "\t" << Bx[NN*(1*6+2)] << "\t" << Bx[NN*(1*6+3)] << "\t" << Bx[NN*(1*6+4)] << "\t" << Bx[NN*(1*6+5)] << std::endl
  //        << Bx[NN*(2*6+0)] << "\t" << Bx[NN*(2*6+1)] << "\t" << Bx[NN*(2*6+2)] << "\t" << Bx[NN*(2*6+3)] << "\t" << Bx[NN*(2*6+4)] << "\t" << Bx[NN*(2*6+5)] << std::endl
  //        << Bx[NN*(3*6+0)] << "\t" << Bx[NN*(3*6+1)] << "\t" << Bx[NN*(3*6+2)] << "\t" << Bx[NN*(3*6+3)] << "\t" << Bx[NN*(3*6+4)] << "\t" << Bx[NN*(3*6+5)] << std::endl
  //        << Bx[NN*(4*6+0)] << "\t" << Bx[NN*(4*6+1)] << "\t" << Bx[NN*(4*6+2)] << "\t" << Bx[NN*(4*6+3)] << "\t" << Bx[NN*(4*6+4)] << "\t" << Bx[NN*(4*6+5)] << std::endl
  //        << Bx[NN*(5*6+0)] << "\t" << Bx[NN*(5*6+1)] << "\t" << Bx[NN*(5*6+2)] << "\t" << Bx[NN*(5*6+3)] << "\t" << Bx[NN*(5*6+4)] << "\t" << Bx[NN*(5*6+5)] << std::endl;
  std::cout << "Cx=" << std::endl
            << Cx[NN*(0*6+0)] << "\t" << Cx[NN*(0*6+1)] << "\t" << Cx[NN*(0*6+2)] << "\t" << Cx[NN*(0*6+3)] << "\t" << Cx[NN*(0*6+4)] << "\t" << Cx[NN*(0*6+5)] << std::endl
            << Cx[NN*(1*6+0)] << "\t" << Cx[NN*(1*6+1)] << "\t" << Cx[NN*(1*6+2)] << "\t" << Cx[NN*(1*6+3)] << "\t" << Cx[NN*(1*6+4)] << "\t" << Cx[NN*(1*6+5)] << std::endl
            << Cx[NN*(2*6+0)] << "\t" << Cx[NN*(2*6+1)] << "\t" << Cx[NN*(2*6+2)] << "\t" << Cx[NN*(2*6+3)] << "\t" << Cx[NN*(2*6+4)] << "\t" << Cx[NN*(2*6+5)] << std::endl
            << Cx[NN*(3*6+0)] << "\t" << Cx[NN*(3*6+1)] << "\t" << Cx[NN*(3*6+2)] << "\t" << Cx[NN*(3*6+3)] << "\t" << Cx[NN*(3*6+4)] << "\t" << Cx[NN*(3*6+5)] << std::endl
            << Cx[NN*(4*6+0)] << "\t" << Cx[NN*(4*6+1)] << "\t" << Cx[NN*(4*6+2)] << "\t" << Cx[NN*(4*6+3)] << "\t" << Cx[NN*(4*6+4)] << "\t" << Cx[NN*(4*6+5)] << std::endl
            << Cx[NN*(5*6+0)] << "\t" << Cx[NN*(5*6+1)] << "\t" << Cx[NN*(5*6+2)] << "\t" << Cx[NN*(5*6+3)] << "\t" << Cx[NN*(5*6+4)] << "\t" << Cx[NN*(5*6+5)] << std::endl;
  float time = float(end-begin)/CLOCKS_PER_SEC;
  std::cout << "plainArray_element -- time for NN=" << NN << " multiplications is " << time << " s, i.e. per track [s]=" << time/float(NN) << std::endl;

  delete Ax, Bx, Cx;
}

void test_plainArray_el16mx(const std::array<xt::xarray<float>, 36>& input, const int type, bool align) {
  //
  float* Ax = (align ? (float*) _mm_malloc(36*NN*sizeof(float), 64) : new float[NN*36]);
  float* Bx = (align ? (float*) _mm_malloc(36*NN*sizeof(float), 64) : new float[NN*36]);
  float* Cx = (align ? (float*) _mm_malloc(36*NN*sizeof(float), 64) : new float[NN*36]);
  for (size_t j=0;j<NN*36;++j) Cx[j]=0.;
  //
  // store in element order for bunches of 16 matrices (a la matriplex)
  for (size_t x=0;x<NN/16;++x) {
    for (size_t i=0;i<36;++i) {
      for (size_t n=0;n<16;++n) {
        Ax[n + i*16 + 16*36*x] = input[i](n+16*x);
        Bx[n + i*16 + 16*36*x] = input[i](n+16*x)+0.5;
      }
    }
  }

  const clock_t begin = clock();
  if (type==1) {
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
  } else {
    for (size_t x = 0; x < NN/16; ++x) {
      const size_t Nx = x*16*36;
      for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
          for (size_t k = 0; k < 6; ++k) {
            for (size_t n = 0; n < 16; ++n) {
              Cx[ Nx + (i*6 + j)*16 + n ] += Ax[ Nx + (i*6 + k)*16 + n ] * Bx[ Nx + (k*6 + j)*16 + n];
            }
          }
        }
      }
    }
  }
  const clock_t end = clock();

  // std::cout << "Ax=" << std::endl
  //        << Ax[16*(0*6+0)] << "\t" << Ax[16*(0*6+1)] << "\t" << Ax[16*(0*6+2)] << "\t" << Ax[16*(0*6+3)] << "\t" << Ax[16*(0*6+4)] << "\t" << Ax[16*(0*6+5)] << std::endl
  //        << Ax[16*(1*6+0)] << "\t" << Ax[16*(1*6+1)] << "\t" << Ax[16*(1*6+2)] << "\t" << Ax[16*(1*6+3)] << "\t" << Ax[16*(1*6+4)] << "\t" << Ax[16*(1*6+5)] << std::endl
  //        << Ax[16*(2*6+0)] << "\t" << Ax[16*(2*6+1)] << "\t" << Ax[16*(2*6+2)] << "\t" << Ax[16*(2*6+3)] << "\t" << Ax[16*(2*6+4)] << "\t" << Ax[16*(2*6+5)] << std::endl
  //        << Ax[16*(3*6+0)] << "\t" << Ax[16*(3*6+1)] << "\t" << Ax[16*(3*6+2)] << "\t" << Ax[16*(3*6+3)] << "\t" << Ax[16*(3*6+4)] << "\t" << Ax[16*(3*6+5)] << std::endl
  //        << Ax[16*(4*6+0)] << "\t" << Ax[16*(4*6+1)] << "\t" << Ax[16*(4*6+2)] << "\t" << Ax[16*(4*6+3)] << "\t" << Ax[16*(4*6+4)] << "\t" << Ax[16*(4*6+5)] << std::endl
  //        << Ax[16*(5*6+0)] << "\t" << Ax[16*(5*6+1)] << "\t" << Ax[16*(5*6+2)] << "\t" << Ax[16*(5*6+3)] << "\t" << Ax[16*(5*6+4)] << "\t" << Ax[16*(5*6+5)] << std::endl;
  // std::cout << "Bx=" << std::endl
  //        << Bx[16*(0*6+0)] << "\t" << Bx[16*(0*6+1)] << "\t" << Bx[16*(0*6+2)] << "\t" << Bx[16*(0*6+3)] << "\t" << Bx[16*(0*6+4)] << "\t" << Bx[16*(0*6+5)] << std::endl
  //        << Bx[16*(1*6+0)] << "\t" << Bx[16*(1*6+1)] << "\t" << Bx[16*(1*6+2)] << "\t" << Bx[16*(1*6+3)] << "\t" << Bx[16*(1*6+4)] << "\t" << Bx[16*(1*6+5)] << std::endl
  //        << Bx[16*(2*6+0)] << "\t" << Bx[16*(2*6+1)] << "\t" << Bx[16*(2*6+2)] << "\t" << Bx[16*(2*6+3)] << "\t" << Bx[16*(2*6+4)] << "\t" << Bx[16*(2*6+5)] << std::endl
  //        << Bx[16*(3*6+0)] << "\t" << Bx[16*(3*6+1)] << "\t" << Bx[16*(3*6+2)] << "\t" << Bx[16*(3*6+3)] << "\t" << Bx[16*(3*6+4)] << "\t" << Bx[16*(3*6+5)] << std::endl
  //        << Bx[16*(4*6+0)] << "\t" << Bx[16*(4*6+1)] << "\t" << Bx[16*(4*6+2)] << "\t" << Bx[16*(4*6+3)] << "\t" << Bx[16*(4*6+4)] << "\t" << Bx[16*(4*6+5)] << std::endl
  //        << Bx[16*(5*6+0)] << "\t" << Bx[16*(5*6+1)] << "\t" << Bx[16*(5*6+2)] << "\t" << Bx[16*(5*6+3)] << "\t" << Bx[16*(5*6+4)] << "\t" << Bx[16*(5*6+5)] << std::endl;
  std::cout << "Cx=" << std::endl
            << Cx[16*(0*6+0)] << "\t" << Cx[16*(0*6+1)] << "\t" << Cx[16*(0*6+2)] << "\t" << Cx[16*(0*6+3)] << "\t" << Cx[16*(0*6+4)] << "\t" << Cx[16*(0*6+5)] << std::endl
            << Cx[16*(1*6+0)] << "\t" << Cx[16*(1*6+1)] << "\t" << Cx[16*(1*6+2)] << "\t" << Cx[16*(1*6+3)] << "\t" << Cx[16*(1*6+4)] << "\t" << Cx[16*(1*6+5)] << std::endl
            << Cx[16*(2*6+0)] << "\t" << Cx[16*(2*6+1)] << "\t" << Cx[16*(2*6+2)] << "\t" << Cx[16*(2*6+3)] << "\t" << Cx[16*(2*6+4)] << "\t" << Cx[16*(2*6+5)] << std::endl
            << Cx[16*(3*6+0)] << "\t" << Cx[16*(3*6+1)] << "\t" << Cx[16*(3*6+2)] << "\t" << Cx[16*(3*6+3)] << "\t" << Cx[16*(3*6+4)] << "\t" << Cx[16*(3*6+5)] << std::endl
            << Cx[16*(4*6+0)] << "\t" << Cx[16*(4*6+1)] << "\t" << Cx[16*(4*6+2)] << "\t" << Cx[16*(4*6+3)] << "\t" << Cx[16*(4*6+4)] << "\t" << Cx[16*(4*6+5)] << std::endl
            << Cx[16*(5*6+0)] << "\t" << Cx[16*(5*6+1)] << "\t" << Cx[16*(5*6+2)] << "\t" << Cx[16*(5*6+3)] << "\t" << Cx[16*(5*6+4)] << "\t" << Cx[16*(5*6+5)] << std::endl;
  float time = float(end-begin)/CLOCKS_PER_SEC;
  if (type==1) std::cout << "plainArray_el16mx (mplex loop) with align=" << align << " -- time for NN=" << NN << " multiplications is " << time << " s, i.e. per track [s]=" << time/float(NN) << std::endl;
  else std::cout << "plainArray_el16mx (plain loop) with align=" << align << " -- time for NN=" << NN << " multiplications is " << time << " s, i.e. per track [s]=" << time/float(NN) << std::endl;
  if (align) {
    _mm_free(Ax);
    _mm_free(Bx);
    _mm_free(Cx);
  } else {
    delete Ax, Bx, Cx;
  }
}


void test_plainArray_xsimd(const std::array<xt::xarray<float>, 36>& input) {
  //
  using vector_type = std::vector<float, XSIMD_DEFAULT_ALLOCATOR(float)>;
  using b_type = xsimd::simd_type<float>;
  //
  vector_type A[36];
  vector_type B[36];
  vector_type C[36];
  //
  for (size_t i=0;i<36;++i) {
    A[i] = vector_type(NN,0.);
    B[i] = vector_type(NN,0.);
    C[i] = vector_type(NN,0.);
    for (size_t x=0;x<NN;++x) {
      A[i][x] = input[i](x);
      B[i][x] = input[i](x)+0.5;
    }
  }
  //
  std::size_t inc = b_type::size;
  std::size_t size = NN;
  //
  // size for which the vectorization is possible (we assume there is nothing left out, i.e. size-vec_size==0)
  std::size_t vec_size = size - size % inc;
  const clock_t begin = clock();
  for(std::size_t i = 0; i < vec_size; i += inc) {
    b_type a[36] = {};
    b_type b[36] = {};
    b_type c[36] = {};
    for (size_t j=0;j<36;++j) {
      a[j] = xsimd::load_aligned(&A[j][i]);
      b[j] = xsimd::load_aligned(&B[j][i]);
    }
    //
    c[ 0] = a[ 0]*b[ 0] + a[ 1]*b[ 6] + a[ 2]*b[12] + a[ 3]*b[18] + a[ 4]*b[24] + a[ 5]*b[30];
    c[ 1] = a[ 0]*b[ 1] + a[ 1]*b[ 7] + a[ 2]*b[13] + a[ 3]*b[19] + a[ 4]*b[25] + a[ 5]*b[31];
    c[ 2] = a[ 0]*b[ 2] + a[ 1]*b[ 8] + a[ 2]*b[14] + a[ 3]*b[20] + a[ 4]*b[26] + a[ 5]*b[32];
    c[ 3] = a[ 0]*b[ 3] + a[ 1]*b[ 9] + a[ 2]*b[15] + a[ 3]*b[21] + a[ 4]*b[27] + a[ 5]*b[33];
    c[ 4] = a[ 0]*b[ 4] + a[ 1]*b[10] + a[ 2]*b[16] + a[ 3]*b[22] + a[ 4]*b[28] + a[ 5]*b[34];
    c[ 5] = a[ 0]*b[ 5] + a[ 1]*b[11] + a[ 2]*b[17] + a[ 3]*b[23] + a[ 4]*b[29] + a[ 5]*b[35];
    c[ 6] = a[ 6]*b[ 0] + a[ 7]*b[ 6] + a[ 8]*b[12] + a[ 9]*b[18] + a[10]*b[24] + a[11]*b[30];
    c[ 7] = a[ 6]*b[ 1] + a[ 7]*b[ 7] + a[ 8]*b[13] + a[ 9]*b[19] + a[10]*b[25] + a[11]*b[31];
    c[ 8] = a[ 6]*b[ 2] + a[ 7]*b[ 8] + a[ 8]*b[14] + a[ 9]*b[20] + a[10]*b[26] + a[11]*b[32];
    c[ 9] = a[ 6]*b[ 3] + a[ 7]*b[ 9] + a[ 8]*b[15] + a[ 9]*b[21] + a[10]*b[27] + a[11]*b[33];
    c[10] = a[ 6]*b[ 4] + a[ 7]*b[10] + a[ 8]*b[16] + a[ 9]*b[22] + a[10]*b[28] + a[11]*b[34];
    c[11] = a[ 6]*b[ 5] + a[ 7]*b[11] + a[ 8]*b[17] + a[ 9]*b[23] + a[10]*b[29] + a[11]*b[35];
    c[12] = a[12]*b[ 0] + a[13]*b[ 6] + a[14]*b[12] + a[15]*b[18] + a[16]*b[24] + a[17]*b[30];
    c[13] = a[12]*b[ 1] + a[13]*b[ 7] + a[14]*b[13] + a[15]*b[19] + a[16]*b[25] + a[17]*b[31];
    c[14] = a[12]*b[ 2] + a[13]*b[ 8] + a[14]*b[14] + a[15]*b[20] + a[16]*b[26] + a[17]*b[32];
    c[15] = a[12]*b[ 3] + a[13]*b[ 9] + a[14]*b[15] + a[15]*b[21] + a[16]*b[27] + a[17]*b[33];
    c[16] = a[12]*b[ 4] + a[13]*b[10] + a[14]*b[16] + a[15]*b[22] + a[16]*b[28] + a[17]*b[34];
    c[17] = a[12]*b[ 5] + a[13]*b[11] + a[14]*b[17] + a[15]*b[23] + a[16]*b[29] + a[17]*b[35];
    c[18] = a[18]*b[ 0] + a[19]*b[ 6] + a[20]*b[12] + a[21]*b[18] + a[22]*b[24] + a[23]*b[30];
    c[19] = a[18]*b[ 1] + a[19]*b[ 7] + a[20]*b[13] + a[21]*b[19] + a[22]*b[25] + a[23]*b[31];
    c[20] = a[18]*b[ 2] + a[19]*b[ 8] + a[20]*b[14] + a[21]*b[20] + a[22]*b[26] + a[23]*b[32];
    c[21] = a[18]*b[ 3] + a[19]*b[ 9] + a[20]*b[15] + a[21]*b[21] + a[22]*b[27] + a[23]*b[33];
    c[22] = a[18]*b[ 4] + a[19]*b[10] + a[20]*b[16] + a[21]*b[22] + a[22]*b[28] + a[23]*b[34];
    c[23] = a[18]*b[ 5] + a[19]*b[11] + a[20]*b[17] + a[21]*b[23] + a[22]*b[29] + a[23]*b[35];
    c[24] = a[24]*b[ 0] + a[25]*b[ 6] + a[26]*b[12] + a[27]*b[18] + a[28]*b[24] + a[29]*b[30];
    c[25] = a[24]*b[ 1] + a[25]*b[ 7] + a[26]*b[13] + a[27]*b[19] + a[28]*b[25] + a[29]*b[31];
    c[26] = a[24]*b[ 2] + a[25]*b[ 8] + a[26]*b[14] + a[27]*b[20] + a[28]*b[26] + a[29]*b[32];
    c[27] = a[24]*b[ 3] + a[25]*b[ 9] + a[26]*b[15] + a[27]*b[21] + a[28]*b[27] + a[29]*b[33];
    c[28] = a[24]*b[ 4] + a[25]*b[10] + a[26]*b[16] + a[27]*b[22] + a[28]*b[28] + a[29]*b[34];
    c[29] = a[24]*b[ 5] + a[25]*b[11] + a[26]*b[17] + a[27]*b[23] + a[28]*b[29] + a[29]*b[35];
    c[30] = a[30]*b[ 0] + a[31]*b[ 6] + a[32]*b[12] + a[33]*b[18] + a[34]*b[24] + a[35]*b[30];
    c[31] = a[30]*b[ 1] + a[31]*b[ 7] + a[32]*b[13] + a[33]*b[19] + a[34]*b[25] + a[35]*b[31];
    c[32] = a[30]*b[ 2] + a[31]*b[ 8] + a[32]*b[14] + a[33]*b[20] + a[34]*b[26] + a[35]*b[32];
    c[33] = a[30]*b[ 3] + a[31]*b[ 9] + a[32]*b[15] + a[33]*b[21] + a[34]*b[27] + a[35]*b[33];
    c[34] = a[30]*b[ 4] + a[31]*b[10] + a[32]*b[16] + a[33]*b[22] + a[34]*b[28] + a[35]*b[34];
    c[35] = a[30]*b[ 5] + a[31]*b[11] + a[32]*b[17] + a[33]*b[23] + a[34]*b[29] + a[35]*b[35];
    //
    for (size_t j=0;j<36;++j) xsimd::store_aligned(&C[j][i],c[j]);
  }
  const clock_t end = clock();
  //
  // std::cout << "A=" << std::endl
  //           << A[(0*6+0)][0] << "\t" << A[(0*6+1)][0] << "\t" << A[(0*6+2)][0] << "\t" << A[(0*6+3)][0] << "\t" << A[(0*6+4)][0] << "\t" << A[(0*6+5)][0] << std::endl
  //           << A[(1*6+0)][0] << "\t" << A[(1*6+1)][0] << "\t" << A[(1*6+2)][0] << "\t" << A[(1*6+3)][0] << "\t" << A[(1*6+4)][0] << "\t" << A[(1*6+5)][0] << std::endl
  //           << A[(2*6+0)][0] << "\t" << A[(2*6+1)][0] << "\t" << A[(2*6+2)][0] << "\t" << A[(2*6+3)][0] << "\t" << A[(2*6+4)][0] << "\t" << A[(2*6+5)][0] << std::endl
  //           << A[(3*6+0)][0] << "\t" << A[(3*6+1)][0] << "\t" << A[(3*6+2)][0] << "\t" << A[(3*6+3)][0] << "\t" << A[(3*6+4)][0] << "\t" << A[(3*6+5)][0] << std::endl
  //           << A[(4*6+0)][0] << "\t" << A[(4*6+1)][0] << "\t" << A[(4*6+2)][0] << "\t" << A[(4*6+3)][0] << "\t" << A[(4*6+4)][0] << "\t" << A[(4*6+5)][0] << std::endl
  //           << A[(5*6+0)][0] << "\t" << A[(5*6+1)][0] << "\t" << A[(5*6+2)][0] << "\t" << A[(5*6+3)][0] << "\t" << A[(5*6+4)][0] << "\t" << A[(5*6+5)][0] << std::endl;
  // std::cout << "B=" << std::endl
  //           << B[(0*6+0)][0] << "\t" << B[(0*6+1)][0] << "\t" << B[(0*6+2)][0] << "\t" << B[(0*6+3)][0] << "\t" << B[(0*6+4)][0] << "\t" << B[(0*6+5)][0] << std::endl
  //           << B[(1*6+0)][0] << "\t" << B[(1*6+1)][0] << "\t" << B[(1*6+2)][0] << "\t" << B[(1*6+3)][0] << "\t" << B[(1*6+4)][0] << "\t" << B[(1*6+5)][0] << std::endl
  //           << B[(2*6+0)][0] << "\t" << B[(2*6+1)][0] << "\t" << B[(2*6+2)][0] << "\t" << B[(2*6+3)][0] << "\t" << B[(2*6+4)][0] << "\t" << B[(2*6+5)][0] << std::endl
  //           << B[(3*6+0)][0] << "\t" << B[(3*6+1)][0] << "\t" << B[(3*6+2)][0] << "\t" << B[(3*6+3)][0] << "\t" << B[(3*6+4)][0] << "\t" << B[(3*6+5)][0] << std::endl
  //           << B[(4*6+0)][0] << "\t" << B[(4*6+1)][0] << "\t" << B[(4*6+2)][0] << "\t" << B[(4*6+3)][0] << "\t" << B[(4*6+4)][0] << "\t" << B[(4*6+5)][0] << std::endl
  //           << B[(5*6+0)][0] << "\t" << B[(5*6+1)][0] << "\t" << B[(5*6+2)][0] << "\t" << B[(5*6+3)][0] << "\t" << B[(5*6+4)][0] << "\t" << B[(5*6+5)][0] << std::endl;
  std::cout << "C=" << std::endl
            << C[(0*6+0)][0] << "\t" << C[(0*6+1)][0] << "\t" << C[(0*6+2)][0] << "\t" << C[(0*6+3)][0] << "\t" << C[(0*6+4)][0] << "\t" << C[(0*6+5)][0] << std::endl
            << C[(1*6+0)][0] << "\t" << C[(1*6+1)][0] << "\t" << C[(1*6+2)][0] << "\t" << C[(1*6+3)][0] << "\t" << C[(1*6+4)][0] << "\t" << C[(1*6+5)][0] << std::endl
            << C[(2*6+0)][0] << "\t" << C[(2*6+1)][0] << "\t" << C[(2*6+2)][0] << "\t" << C[(2*6+3)][0] << "\t" << C[(2*6+4)][0] << "\t" << C[(2*6+5)][0] << std::endl
            << C[(3*6+0)][0] << "\t" << C[(3*6+1)][0] << "\t" << C[(3*6+2)][0] << "\t" << C[(3*6+3)][0] << "\t" << C[(3*6+4)][0] << "\t" << C[(3*6+5)][0] << std::endl
            << C[(4*6+0)][0] << "\t" << C[(4*6+1)][0] << "\t" << C[(4*6+2)][0] << "\t" << C[(4*6+3)][0] << "\t" << C[(4*6+4)][0] << "\t" << C[(4*6+5)][0] << std::endl
            << C[(5*6+0)][0] << "\t" << C[(5*6+1)][0] << "\t" << C[(5*6+2)][0] << "\t" << C[(5*6+3)][0] << "\t" << C[(5*6+4)][0] << "\t" << C[(5*6+5)][0] << std::endl;
  float time = float(end-begin)/CLOCKS_PER_SEC;
  std::cout << "plainArray_xsimd -- time for NN=" << NN << " multiplications is " << time << " s, i.e. per track [s]=" << time/float(NN) << std::endl;
}

int main(int argc, char* argv[])
{

  std::array<xt::xarray<float>, 36> input;
  for (size_t i=0;i<36;++i) input[i] = xt::linspace<float>(i,i+100,NN);

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
  test_v3(input,0);
  std::cout << std::endl;

  test_plainArray_matrix(input,0);
  std::cout << std::endl;
  test_plainArray_element(input,0);
  std::cout << std::endl;
  test_plainArray_el16mx(input,0,0);
  std::cout << std::endl;
  test_plainArray_el16mx(input,1,0);
  std::cout << std::endl;
  test_plainArray_el16mx(input,1,1);
  std::cout << std::endl;
  test_plainArray_xsimd(input);

  return 0;
}
