#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include <time.h>

constexpr size_t NN = 16*600000;//10000000;
constexpr size_t nrep = 10;

#include "MatriplXT.h"

// /usr/local/Cellar/gcc/7.1.0/bin/c++-7 -O2 -mavx -DXTENSOR_USE_XSIMD -I /Users/cerati/miniconda3/include/ xtensor-test.cc -o xtensor-test.exe

void test_v0(const std::array<xt::xarray<float>, 36>& input, const int type) {

  MatriplXT66 A;
  for (size_t i=0;i<36;++i) A[i] = input[i];

  MatriplXT66 B;
  for (size_t i=0;i<36;++i) B[i] = input[i]+0.5;

  MatriplXT66 C;
  for (size_t i=0;i<36;++i) C[i] = xt::zeros<float>({NN});

  //dry run, not timed
  if (type==1) MultiplyXT66Loop(A, B, C);
  else if (type==2) MultiplyXT66LoopTile(A, B, C);
  else MultiplyXT66(A, B, C);

  //timed run, repeated nrep times
  const clock_t begin = clock();
  for (size_t nit=0; nit<nrep; ++nit) {
    if (type==1) MultiplyXT66Loop(A, B, C);
    else if (type==2) MultiplyXT66LoopTile(A, B, C);
    else MultiplyXT66(A, B, C);
  }
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
  if (type==1) std::cout << "v0 -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s (loop version), i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
  else if (type==2) std::cout << "v0 -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s (loop-tile version), i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
  else std::cout << "v0 -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s, i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
}

void test_v1(const std::array<xt::xarray<float>, 36>& input, const int type) {

  MatriplXT66_v1 A(NN);
  for (size_t i=0;i<36;++i) A[i] = input[i];

  MatriplXT66_v1 B(NN);
  for (size_t i=0;i<36;++i) B[i] = input[i]+0.5;

  MatriplXT66_v1 C(NN);
  for (size_t i=0;i<36;++i) C[i] = xt::zeros<float>({NN});

  //dry run, not timed
  if (type==1) MultiplyXT66Loop(A, B, C);
  else if (type==2) MultiplyXT66Stack(A, B, C);
  else MultiplyXT66(A, B, C);

  //timed run, repeated nrep times
  const clock_t begin = clock();
  for (size_t nit=0; nit<nrep; ++nit) {
    if (type==1) MultiplyXT66Loop(A, B, C);
    else if (type==2) MultiplyXT66Stack(A, B, C);
    else MultiplyXT66(A, B, C);
  }
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
  if (type==1) std::cout << "v1 -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s (loop version), i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
  else if (type==2) std::cout << "v1 -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s (stack version), i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
  else std::cout << "v1 -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s, i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
}

void test_v2(const std::array<xt::xarray<float>, 36>& input, const int type) {

  MatriplXT66_v2 A(NN);
  for (size_t i=0;i<36;++i) A[i] = input[i];

  MatriplXT66_v2 B(NN);
  for (size_t i=0;i<36;++i) B[i] = input[i]+0.5;

  MatriplXT66_v2 C(NN);
  for (size_t i=0;i<36;++i) C[i] = xt::zeros<float>({NN});

  //dry run, not timed
  if (type==1) MultiplyXT66Loop(A, B, C);
  else if (type==2) MultiplyXT66Stack(A, B, C);
  else MultiplyXT66(A, B, C);

  //timed run, repeated nrep times
  const clock_t begin = clock();
  for (size_t nit=0; nit<nrep; ++nit) {
    if (type==1) MultiplyXT66Loop(A, B, C);
    else if (type==2) MultiplyXT66Stack(A, B, C);
    else MultiplyXT66(A, B, C);
  }
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
  if (type==1) std::cout << "v2 -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s (loop version), i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
  else if (type==2) std::cout << "v2 -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s (stack version), i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
  else std::cout << "v2 -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s, i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
}

void test_v3(const std::array<xt::xarray<float>, 36>& input, const int type) {

  MatriplXT66_v3 A;
  for (size_t i=0;i<36;++i) A[i] = input[i];

  MatriplXT66_v3 B;
  for (size_t i=0;i<36;++i) B[i] = input[i]+0.5;

  MatriplXT66_v3 C;
  for (size_t i=0;i<36;++i) C[i] = xt::zeros<float>({NN});

  //dry run, not timed
  MultiplyXT66(A, B, C);

  //timed run, repeated nrep times
  const clock_t begin = clock();
  for (size_t nit=0; nit<nrep; ++nit) {
    MultiplyXT66(A, B, C);
  }
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
  // if (type==1) std::cout << "v3 -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s (loop version), i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
  // else if (type==2) std::cout << "v3 -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s (stack version), i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
  // else
  std::cout << "v3 -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s, i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
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

  //dry run, not timed
  plainArray_matrix_mult66(Ax,Bx,Cx);

  //timed run, repeated nrep times
  const clock_t begin = clock();
  for (size_t nit=0; nit<nrep; ++nit) {
    plainArray_matrix_mult66(Ax,Bx,Cx);
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
  std::cout << "plainArray_matrix -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s, i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;

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

  //dry run, not timed
  plainArray_element_mult66(Ax,Bx,Cx);

  //timed run, repeated nrep times
  const clock_t begin = clock();
  for (size_t nit=0; nit<nrep; ++nit) {
    plainArray_element_mult66(Ax,Bx,Cx);
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
  std::cout << "plainArray_element -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s, i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;

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

  //dry run, not timed
  if (type==1) {
    plainArray_el16mx_mult66_v1(Ax,Bx,Cx);
  } else {
    plainArray_el16mx_mult66(Ax,Bx,Cx);
  }

  //timed run, repeated nrep times
  const clock_t begin = clock();
  for (size_t nit=0; nit<nrep; ++nit) {
    if (type==1) {
      plainArray_el16mx_mult66_v1(Ax,Bx,Cx);
    } else {
      plainArray_el16mx_mult66(Ax,Bx,Cx);
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
  if (type==1) std::cout << "plainArray_el16mx (mplex loop) with align=" << align << " -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s, i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
  else std::cout << "plainArray_el16mx (plain loop) with align=" << align << " -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s, i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
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
  const std::size_t inc = b_type::size;
  const std::size_t size = NN;
  //
  // size for which the vectorization is possible (we assume there is nothing left out, i.e. size-vec_size==0)
  const std::size_t vec_size = size - size % inc;

  //dry run, not timed
  plainArray_xsimd_mult66(A, B, C, vec_size, inc);

  //timed run, repeated nrep times
  const clock_t begin = clock();
  for (size_t nit=0; nit<nrep; ++nit) {
    plainArray_xsimd_mult66(A, B, C, vec_size, inc);
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
  std::cout << "plainArray_xsimd -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s, i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
}

int main(int argc, char* argv[])
{

  std::cout << "running with NN=" << NN << " nrep=" << nrep << std::endl;

  std::array<xt::xarray<float>, 36> input;
  for (size_t i=0;i<36;++i) input[i] = xt::linspace<float>(i,i+100,NN);

  std::cout << "done preparing input" << std::endl;

  test_v0(input,0);
  // test_v0(input,1);
  // test_v0(input,2);
  std::cout << std::endl;
  test_v1(input,0);
  // test_v1(input,1);
  // // test_v1(input,2);
  std::cout << std::endl;
  test_v2(input,0);
  // test_v2(input,1);
  // // test_v2(input,2);
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
