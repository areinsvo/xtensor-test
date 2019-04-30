#include <iostream>
#include <time.h>
#include <array>

#include "Matriplex.h"

//constexpr size_t NN = 16*3000;//1000
constexpr int NN = 9600000;//10000000;
constexpr size_t nrep = 12;


// /usr/local/Cellar/gcc/7.1.0/bin/c++-7 -O2 -mavx -DXTENSOR_USE_XSIMD -I /Users/cerati/miniconda3/include/ xtensor-test.cc -o xtensor-test.exe
void test_plainArray_matrix_par(const float input[][NN], const int type) {
  float* Ax = new float[NN*36];
  float* Bx = new float[NN*36];
  float* Cx = new float[NN*36];
  for (size_t j=0;j<NN*36;++j) Cx[j]=0.;
  // store in matrix order (i.e. all elements of a given matrix are contiguous) 
  for (size_t x=0;x<NN;++x) {
    for (size_t i=0;i<36;++i) {
      Ax[i + 36*x] = input[i][x];
      Bx[i + 36*x] = input[i][x]+0.5;
    }
  }

  //dry run, not timed 
  plainArray_matrix_mult66_par(Ax,Bx,Cx,NN);

  //timed run, repeated nrep times
  const clock_t begin = clock();
  double tdata = omp_get_wtime();
#pragma omp parallel for
  for (size_t nit=0; nit<nrep; ++nit) {
    plainArray_matrix_mult66_par(Ax,Bx,Cx,NN);
  }
  tdata = omp_get_wtime() - tdata;
  const clock_t end = clock();
  std::cout << "Cx=" << std::endl
            << Cx[(0*6+0)] << "\t" << Cx[(0*6+1)] << "\t" << Cx[(0*6+2)] << "\t" << Cx[(0*6+3)] << "\t" << Cx[(0*6+4)] << "\t" << Cx[(0*6+5)] << std::endl
            << Cx[(1*6+0)] << "\t" << Cx[(1*6+1)] << "\t" << Cx[(1*6+2)] << "\t" << Cx[(1*6+3)] << "\t" << Cx[(1*6+4)] << "\t" << Cx[(1*6+5)] << std::endl
            << Cx[(2*6+0)] << "\t" << Cx[(2*6+1)] << "\t" << Cx[(2*6+2)] << "\t" << Cx[(2*6+3)] << "\t" << Cx[(2*6+4)] << "\t" << Cx[(2*6+5)] << std::endl
            << Cx[(3*6+0)] << "\t" << Cx[(3*6+1)] << "\t" << Cx[(3*6+2)] << "\t" << Cx[(3*6+3)] << "\t" << Cx[(3*6+4)] << "\t" << Cx[(3*6+5)] << std::endl
            << Cx[(4*6+0)] << "\t" << Cx[(4*6+1)] << "\t" << Cx[(4*6+2)] << "\t" << Cx[(4*6+3)] << "\t" << Cx[(4*6+4)] << "\t" << Cx[(4*6+5)] << std::endl
            << Cx[(5*6+0)] << "\t" << Cx[(5*6+1)] << "\t" << Cx[(5*6+2)] << "\t" << Cx[(5*6+3)] << "\t" << Cx[(5*6+4)] << "\t" << Cx[(5*6+5)] << std::endl;
  float time = float(end-begin)/CLOCKS_PER_SEC;
  std::cout << "plainArray_matrix_par -- tdata for NN*nrep=" << NN*nrep << " multiplications is " << tdata << " s, i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;
  std::cout << "plainArray_matrix_par -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s, i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;

  delete Ax, Bx, Cx;
}

void test_plainArray_matrix(const float input[][NN], const int type) {
  //void test_plainArray_matrix(const std::array<std::array<float,NN>, 36>& input, const int type) {
  //
  float* Ax = new float[NN*36];
  float* Bx = new float[NN*36];
  float* Cx = new float[NN*36];
  for (size_t j=0;j<NN*36;++j) Cx[j]=0.;
  //
  // store in matrix order (i.e. all elements of a given matrix are contiguous)
  for (size_t x=0;x<NN;++x) {
    for (size_t i=0;i<36;++i) {
      Ax[i + 36*x] = input[i][x];
      Bx[i + 36*x] = input[i][x]+0.5;
    }
  }

  //dry run, not timed
  plainArray_matrix_mult66(Ax,Bx,Cx,NN);

  //timed run, repeated nrep times
  const clock_t begin = clock();
  double tdata = omp_get_wtime();
  for (size_t nit=0; nit<nrep; ++nit) {
    plainArray_matrix_mult66(Ax,Bx,Cx,NN);
  }
  tdata = omp_get_wtime() - tdata;
  const clock_t end = clock();

  std::cout << "Cx=" << std::endl
            << Cx[(0*6+0)] << "\t" << Cx[(0*6+1)] << "\t" << Cx[(0*6+2)] << "\t" << Cx[(0*6+3)] << "\t" << Cx[(0*6+4)] << "\t" << Cx[(0*6+5)] << std::endl
            << Cx[(1*6+0)] << "\t" << Cx[(1*6+1)] << "\t" << Cx[(1*6+2)] << "\t" << Cx[(1*6+3)] << "\t" << Cx[(1*6+4)] << "\t" << Cx[(1*6+5)] << std::endl
            << Cx[(2*6+0)] << "\t" << Cx[(2*6+1)] << "\t" << Cx[(2*6+2)] << "\t" << Cx[(2*6+3)] << "\t" << Cx[(2*6+4)] << "\t" << Cx[(2*6+5)] << std::endl
            << Cx[(3*6+0)] << "\t" << Cx[(3*6+1)] << "\t" << Cx[(3*6+2)] << "\t" << Cx[(3*6+3)] << "\t" << Cx[(3*6+4)] << "\t" << Cx[(3*6+5)] << std::endl
            << Cx[(4*6+0)] << "\t" << Cx[(4*6+1)] << "\t" << Cx[(4*6+2)] << "\t" << Cx[(4*6+3)] << "\t" << Cx[(4*6+4)] << "\t" << Cx[(4*6+5)] << std::endl
            << Cx[(5*6+0)] << "\t" << Cx[(5*6+1)] << "\t" << Cx[(5*6+2)] << "\t" << Cx[(5*6+3)] << "\t" << Cx[(5*6+4)] << "\t" << Cx[(5*6+5)] << std::endl;
  float time = float(end-begin)/CLOCKS_PER_SEC;
  std::cout << "plainArray_matrix -- tdata for NN*nrep=" << NN*nrep << " multiplications is " << tdata << " s, i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;

  std::cout << "plainArray_matrix -- time for NN*nrep=" << NN*nrep << " multiplications is " << time << " s, i.e. per mult. [s]=" << time/float(NN*nrep) << std::endl;

  delete Ax, Bx, Cx;
}

void test_plainArray_element(const float  input[][NN], const int type) {
  //void test_plainArray_element(const std::array<std::array<float,NN>, 36>& input, const int type) {
  //
  float* Ax = new float[NN*36];
  float* Bx = new float[NN*36];
  float* Cx = new float[NN*36];
  for (size_t j=0;j<NN*36;++j) Cx[j]=0.;
  //
  // store in element order (i.e. all matrices for a given element are contiguous)
  for (size_t x=0;x<NN;++x) {
    for (size_t i=0;i<36;++i) {
      Ax[i*NN + x] = input[i][x];
      Bx[i*NN + x] = input[i][x]+0.5;
    }
  }

  //dry run, not timed
  plainArray_element_mult66(Ax,Bx,Cx,NN);

  //timed run, repeated nrep times
  const clock_t begin = clock();
  for (size_t nit=0; nit<nrep; ++nit) {
    plainArray_element_mult66(Ax,Bx,Cx,NN);
  }
  const clock_t end = clock();


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
void test_plainArray_el16mx(const float input[][NN], const int type, bool align) {
  //void test_plainArray_el16mx(const std::array<std::array<float,NN>, 36>& input, const int type, bool align) {
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
        Ax[n + i*16 + 16*36*x] = input[i][n+16*x];
        Bx[n + i*16 + 16*36*x] = input[i][n+16*x]+0.5;
      }
    }
  }

  //dry run, not timed
  if (type==1) {
    plainArray_el16mx_mult66_v1(Ax,Bx,Cx,NN);
  } else {
    plainArray_el16mx_mult66(Ax,Bx,Cx,NN);
  }

  //timed run, repeated nrep times
  const clock_t begin = clock();
  for (size_t nit=0; nit<nrep; ++nit) {
    if (type==1) {
      plainArray_el16mx_mult66_v1(Ax,Bx,Cx,NN);
    } else {
      plainArray_el16mx_mult66(Ax,Bx,Cx,NN);
    }
  }
  const clock_t end = clock();

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


float input[36][NN];
int main(int argc, char* argv[])
{

  std::cout << "running with NN=" << NN << " nrep=" << nrep << std::endl;

  //  std::array<std::array<float, NN>, 36> input;
  for (size_t i=0;i<36;++i)
    {
      //      input[i] = {};
      //      std::cout << input[i].size() << ", " << NN << std::endl;
      for(size_t j =0; j<NN;++j) //16 here seems to work, 17 does not, even though NN is set to 9600000
        {
	  //	  std::cout << "Hi" << std::endl;
	  //	  std::cout << input[i][j] << std::endl;
	  //	  std::cout << input[i].size() << ", "<< NN << ", " << input[i][17] << std::endl;
	  //std::cout << i+(100.0/NN)*j << ", ";
	  float step_size = 0.1;//std::max(0.1,100.0/NN);
	  input[i][j] = i+step_size*j;
	}
    }
  std::cout << "done preparing input" << std::endl;
  
  test_plainArray_matrix(input,0);
  std::cout << std::endl;
  test_plainArray_matrix_par(input,0);
  std::cout << std::endl;
  test_plainArray_element(input,0);
  std::cout << std::endl;
  test_plainArray_el16mx(input,0,0);
  std::cout << std::endl;
  test_plainArray_el16mx(input,1,0);
  std::cout << std::endl;
  test_plainArray_el16mx(input,1,1);
  std::cout << std::endl;
  
  return 0;
}
