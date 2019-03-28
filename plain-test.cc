#include <iostream>
#include <time.h>
#include <caliper/cali.h>
#include <stdlib.h>
#include <stdio.h>
#include <array>
#include <random>
// /usr/local/Cellar/gcc/7.1.0/bin/c++-7 -O2 -mavx -DXTENSOR_USE_XSIMD -I /Users/cerati/miniconda3/include/ xtensor-test.cc -o xtensor-test.exe

constexpr size_t NN = 16*600000;//10000000;

struct alignas(64) vec_t
{
    float data[NN*36];
};

void test_plain(std::array<std::array<float,NN>,36>& input) {
  //
  vec_t* Ax = new vec_t;
  vec_t* Bx = new vec_t;
  vec_t* Cx = new vec_t;
  for (size_t j=0;j<NN*36;++j) Cx->data[j]=0.;
  //
  for (size_t x=0;x<NN/16;++x) {
    for (size_t i=0;i<36;++i) {
      for (size_t n=0;n<16;++n) {
        Ax->data[n + i*16 + 16*36*x] = input[i][n+16*x];
        Bx->data[n + i*16 + 16*36*x] = input[i][n+16*x];
      }
    }
  }

  const clock_t begin = clock();
  for (size_t x = 0; x < NN/16; ++x) {
    const size_t Nx = x*16*36;
    for (size_t n = 0; n < 16; ++n) {
      Cx->data[Nx+16* 0+n] = Ax->data[Nx+16* 0+n]*Bx->data[Nx+16* 0+n] + Ax->data[Nx+16* 1+n]*Bx->data[Nx+16* 6+n] + Ax->data[Nx+16* 2+n]*Bx->data[Nx+16*12+n] + Ax->data[Nx+16* 3+n]*Bx->data[Nx+16*18+n] + Ax->data[Nx+16* 4+n]*Bx->data[Nx+16*24+n] + Ax->data[Nx+16* 5+n]*Bx->data[Nx+16*30+n];
      Cx->data[Nx+16* 1+n] = Ax->data[Nx+16* 0+n]*Bx->data[Nx+16* 1+n] + Ax->data[Nx+16* 1+n]*Bx->data[Nx+16* 7+n] + Ax->data[Nx+16* 2+n]*Bx->data[Nx+16*13+n] + Ax->data[Nx+16* 3+n]*Bx->data[Nx+16*19+n] + Ax->data[Nx+16* 4+n]*Bx->data[Nx+16*25+n] + Ax->data[Nx+16* 5+n]*Bx->data[Nx+16*31+n];
      Cx->data[Nx+16* 2+n] = Ax->data[Nx+16* 0+n]*Bx->data[Nx+16* 2+n] + Ax->data[Nx+16* 1+n]*Bx->data[Nx+16* 8+n] + Ax->data[Nx+16* 2+n]*Bx->data[Nx+16*14+n] + Ax->data[Nx+16* 3+n]*Bx->data[Nx+16*20+n] + Ax->data[Nx+16* 4+n]*Bx->data[Nx+16*26+n] + Ax->data[Nx+16* 5+n]*Bx->data[Nx+16*32+n];
      Cx->data[Nx+16* 3+n] = Ax->data[Nx+16* 0+n]*Bx->data[Nx+16* 3+n] + Ax->data[Nx+16* 1+n]*Bx->data[Nx+16* 9+n] + Ax->data[Nx+16* 2+n]*Bx->data[Nx+16*15+n] + Ax->data[Nx+16* 3+n]*Bx->data[Nx+16*21+n] + Ax->data[Nx+16* 4+n]*Bx->data[Nx+16*27+n] + Ax->data[Nx+16* 5+n]*Bx->data[Nx+16*33+n];
      Cx->data[Nx+16* 4+n] = Ax->data[Nx+16* 0+n]*Bx->data[Nx+16* 4+n] + Ax->data[Nx+16* 1+n]*Bx->data[Nx+16*10+n] + Ax->data[Nx+16* 2+n]*Bx->data[Nx+16*16+n] + Ax->data[Nx+16* 3+n]*Bx->data[Nx+16*22+n] + Ax->data[Nx+16* 4+n]*Bx->data[Nx+16*28+n] + Ax->data[Nx+16* 5+n]*Bx->data[Nx+16*34+n];
      Cx->data[Nx+16* 5+n] = Ax->data[Nx+16* 0+n]*Bx->data[Nx+16* 5+n] + Ax->data[Nx+16* 1+n]*Bx->data[Nx+16*11+n] + Ax->data[Nx+16* 2+n]*Bx->data[Nx+16*17+n] + Ax->data[Nx+16* 3+n]*Bx->data[Nx+16*23+n] + Ax->data[Nx+16* 4+n]*Bx->data[Nx+16*29+n] + Ax->data[Nx+16* 5+n]*Bx->data[Nx+16*35+n];
      Cx->data[Nx+16* 6+n] = Ax->data[Nx+16* 6+n]*Bx->data[Nx+16* 0+n] + Ax->data[Nx+16* 7+n]*Bx->data[Nx+16* 6+n] + Ax->data[Nx+16* 8+n]*Bx->data[Nx+16*12+n] + Ax->data[Nx+16* 9+n]*Bx->data[Nx+16*18+n] + Ax->data[Nx+16*10+n]*Bx->data[Nx+16*24+n] + Ax->data[Nx+16*11+n]*Bx->data[Nx+16*30+n];
      Cx->data[Nx+16* 7+n] = Ax->data[Nx+16* 6+n]*Bx->data[Nx+16* 1+n] + Ax->data[Nx+16* 7+n]*Bx->data[Nx+16* 7+n] + Ax->data[Nx+16* 8+n]*Bx->data[Nx+16*13+n] + Ax->data[Nx+16* 9+n]*Bx->data[Nx+16*19+n] + Ax->data[Nx+16*10+n]*Bx->data[Nx+16*25+n] + Ax->data[Nx+16*11+n]*Bx->data[Nx+16*31+n];
      Cx->data[Nx+16* 8+n] = Ax->data[Nx+16* 6+n]*Bx->data[Nx+16* 2+n] + Ax->data[Nx+16* 7+n]*Bx->data[Nx+16* 8+n] + Ax->data[Nx+16* 8+n]*Bx->data[Nx+16*14+n] + Ax->data[Nx+16* 9+n]*Bx->data[Nx+16*20+n] + Ax->data[Nx+16*10+n]*Bx->data[Nx+16*26+n] + Ax->data[Nx+16*11+n]*Bx->data[Nx+16*32+n];
      Cx->data[Nx+16* 9+n] = Ax->data[Nx+16* 6+n]*Bx->data[Nx+16* 3+n] + Ax->data[Nx+16* 7+n]*Bx->data[Nx+16* 9+n] + Ax->data[Nx+16* 8+n]*Bx->data[Nx+16*15+n] + Ax->data[Nx+16* 9+n]*Bx->data[Nx+16*21+n] + Ax->data[Nx+16*10+n]*Bx->data[Nx+16*27+n] + Ax->data[Nx+16*11+n]*Bx->data[Nx+16*33+n];
      Cx->data[Nx+16*10+n] = Ax->data[Nx+16* 6+n]*Bx->data[Nx+16* 4+n] + Ax->data[Nx+16* 7+n]*Bx->data[Nx+16*10+n] + Ax->data[Nx+16* 8+n]*Bx->data[Nx+16*16+n] + Ax->data[Nx+16* 9+n]*Bx->data[Nx+16*22+n] + Ax->data[Nx+16*10+n]*Bx->data[Nx+16*28+n] + Ax->data[Nx+16*11+n]*Bx->data[Nx+16*34+n];
      Cx->data[Nx+16*11+n] = Ax->data[Nx+16* 6+n]*Bx->data[Nx+16* 5+n] + Ax->data[Nx+16* 7+n]*Bx->data[Nx+16*11+n] + Ax->data[Nx+16* 8+n]*Bx->data[Nx+16*17+n] + Ax->data[Nx+16* 9+n]*Bx->data[Nx+16*23+n] + Ax->data[Nx+16*10+n]*Bx->data[Nx+16*29+n] + Ax->data[Nx+16*11+n]*Bx->data[Nx+16*35+n];
      Cx->data[Nx+16*12+n] = Ax->data[Nx+16*12+n]*Bx->data[Nx+16* 0+n] + Ax->data[Nx+16*13+n]*Bx->data[Nx+16* 6+n] + Ax->data[Nx+16*14+n]*Bx->data[Nx+16*12+n] + Ax->data[Nx+16*15+n]*Bx->data[Nx+16*18+n] + Ax->data[Nx+16*16+n]*Bx->data[Nx+16*24+n] + Ax->data[Nx+16*17+n]*Bx->data[Nx+16*30+n];
      Cx->data[Nx+16*13+n] = Ax->data[Nx+16*12+n]*Bx->data[Nx+16* 1+n] + Ax->data[Nx+16*13+n]*Bx->data[Nx+16* 7+n] + Ax->data[Nx+16*14+n]*Bx->data[Nx+16*13+n] + Ax->data[Nx+16*15+n]*Bx->data[Nx+16*19+n] + Ax->data[Nx+16*16+n]*Bx->data[Nx+16*25+n] + Ax->data[Nx+16*17+n]*Bx->data[Nx+16*31+n];
      Cx->data[Nx+16*14+n] = Ax->data[Nx+16*12+n]*Bx->data[Nx+16* 2+n] + Ax->data[Nx+16*13+n]*Bx->data[Nx+16* 8+n] + Ax->data[Nx+16*14+n]*Bx->data[Nx+16*14+n] + Ax->data[Nx+16*15+n]*Bx->data[Nx+16*20+n] + Ax->data[Nx+16*16+n]*Bx->data[Nx+16*26+n] + Ax->data[Nx+16*17+n]*Bx->data[Nx+16*32+n];
      Cx->data[Nx+16*15+n] = Ax->data[Nx+16*12+n]*Bx->data[Nx+16* 3+n] + Ax->data[Nx+16*13+n]*Bx->data[Nx+16* 9+n] + Ax->data[Nx+16*14+n]*Bx->data[Nx+16*15+n] + Ax->data[Nx+16*15+n]*Bx->data[Nx+16*21+n] + Ax->data[Nx+16*16+n]*Bx->data[Nx+16*27+n] + Ax->data[Nx+16*17+n]*Bx->data[Nx+16*33+n];
      Cx->data[Nx+16*16+n] = Ax->data[Nx+16*12+n]*Bx->data[Nx+16* 4+n] + Ax->data[Nx+16*13+n]*Bx->data[Nx+16*10+n] + Ax->data[Nx+16*14+n]*Bx->data[Nx+16*16+n] + Ax->data[Nx+16*15+n]*Bx->data[Nx+16*22+n] + Ax->data[Nx+16*16+n]*Bx->data[Nx+16*28+n] + Ax->data[Nx+16*17+n]*Bx->data[Nx+16*34+n];
      Cx->data[Nx+16*17+n] = Ax->data[Nx+16*12+n]*Bx->data[Nx+16* 5+n] + Ax->data[Nx+16*13+n]*Bx->data[Nx+16*11+n] + Ax->data[Nx+16*14+n]*Bx->data[Nx+16*17+n] + Ax->data[Nx+16*15+n]*Bx->data[Nx+16*23+n] + Ax->data[Nx+16*16+n]*Bx->data[Nx+16*29+n] + Ax->data[Nx+16*17+n]*Bx->data[Nx+16*35+n];
      Cx->data[Nx+16*18+n] = Ax->data[Nx+16*18+n]*Bx->data[Nx+16* 0+n] + Ax->data[Nx+16*19+n]*Bx->data[Nx+16* 6+n] + Ax->data[Nx+16*20+n]*Bx->data[Nx+16*12+n] + Ax->data[Nx+16*21+n]*Bx->data[Nx+16*18+n] + Ax->data[Nx+16*22+n]*Bx->data[Nx+16*24+n] + Ax->data[Nx+16*23+n]*Bx->data[Nx+16*30+n];
      Cx->data[Nx+16*19+n] = Ax->data[Nx+16*18+n]*Bx->data[Nx+16* 1+n] + Ax->data[Nx+16*19+n]*Bx->data[Nx+16* 7+n] + Ax->data[Nx+16*20+n]*Bx->data[Nx+16*13+n] + Ax->data[Nx+16*21+n]*Bx->data[Nx+16*19+n] + Ax->data[Nx+16*22+n]*Bx->data[Nx+16*25+n] + Ax->data[Nx+16*23+n]*Bx->data[Nx+16*31+n];
      Cx->data[Nx+16*20+n] = Ax->data[Nx+16*18+n]*Bx->data[Nx+16* 2+n] + Ax->data[Nx+16*19+n]*Bx->data[Nx+16* 8+n] + Ax->data[Nx+16*20+n]*Bx->data[Nx+16*14+n] + Ax->data[Nx+16*21+n]*Bx->data[Nx+16*20+n] + Ax->data[Nx+16*22+n]*Bx->data[Nx+16*26+n] + Ax->data[Nx+16*23+n]*Bx->data[Nx+16*32+n];
      Cx->data[Nx+16*21+n] = Ax->data[Nx+16*18+n]*Bx->data[Nx+16* 3+n] + Ax->data[Nx+16*19+n]*Bx->data[Nx+16* 9+n] + Ax->data[Nx+16*20+n]*Bx->data[Nx+16*15+n] + Ax->data[Nx+16*21+n]*Bx->data[Nx+16*21+n] + Ax->data[Nx+16*22+n]*Bx->data[Nx+16*27+n] + Ax->data[Nx+16*23+n]*Bx->data[Nx+16*33+n];
      Cx->data[Nx+16*22+n] = Ax->data[Nx+16*18+n]*Bx->data[Nx+16* 4+n] + Ax->data[Nx+16*19+n]*Bx->data[Nx+16*10+n] + Ax->data[Nx+16*20+n]*Bx->data[Nx+16*16+n] + Ax->data[Nx+16*21+n]*Bx->data[Nx+16*22+n] + Ax->data[Nx+16*22+n]*Bx->data[Nx+16*28+n] + Ax->data[Nx+16*23+n]*Bx->data[Nx+16*34+n];
      Cx->data[Nx+16*23+n] = Ax->data[Nx+16*18+n]*Bx->data[Nx+16* 5+n] + Ax->data[Nx+16*19+n]*Bx->data[Nx+16*11+n] + Ax->data[Nx+16*20+n]*Bx->data[Nx+16*17+n] + Ax->data[Nx+16*21+n]*Bx->data[Nx+16*23+n] + Ax->data[Nx+16*22+n]*Bx->data[Nx+16*29+n] + Ax->data[Nx+16*23+n]*Bx->data[Nx+16*35+n];
      Cx->data[Nx+16*24+n] = Ax->data[Nx+16*24+n]*Bx->data[Nx+16* 0+n] + Ax->data[Nx+16*25+n]*Bx->data[Nx+16* 6+n] + Ax->data[Nx+16*26+n]*Bx->data[Nx+16*12+n] + Ax->data[Nx+16*27+n]*Bx->data[Nx+16*18+n] + Ax->data[Nx+16*28+n]*Bx->data[Nx+16*24+n] + Ax->data[Nx+16*29+n]*Bx->data[Nx+16*30+n];
      Cx->data[Nx+16*25+n] = Ax->data[Nx+16*24+n]*Bx->data[Nx+16* 1+n] + Ax->data[Nx+16*25+n]*Bx->data[Nx+16* 7+n] + Ax->data[Nx+16*26+n]*Bx->data[Nx+16*13+n] + Ax->data[Nx+16*27+n]*Bx->data[Nx+16*19+n] + Ax->data[Nx+16*28+n]*Bx->data[Nx+16*25+n] + Ax->data[Nx+16*29+n]*Bx->data[Nx+16*31+n];
      Cx->data[Nx+16*26+n] = Ax->data[Nx+16*24+n]*Bx->data[Nx+16* 2+n] + Ax->data[Nx+16*25+n]*Bx->data[Nx+16* 8+n] + Ax->data[Nx+16*26+n]*Bx->data[Nx+16*14+n] + Ax->data[Nx+16*27+n]*Bx->data[Nx+16*20+n] + Ax->data[Nx+16*28+n]*Bx->data[Nx+16*26+n] + Ax->data[Nx+16*29+n]*Bx->data[Nx+16*32+n];
      Cx->data[Nx+16*27+n] = Ax->data[Nx+16*24+n]*Bx->data[Nx+16* 3+n] + Ax->data[Nx+16*25+n]*Bx->data[Nx+16* 9+n] + Ax->data[Nx+16*26+n]*Bx->data[Nx+16*15+n] + Ax->data[Nx+16*27+n]*Bx->data[Nx+16*21+n] + Ax->data[Nx+16*28+n]*Bx->data[Nx+16*27+n] + Ax->data[Nx+16*29+n]*Bx->data[Nx+16*33+n];
      Cx->data[Nx+16*28+n] = Ax->data[Nx+16*24+n]*Bx->data[Nx+16* 4+n] + Ax->data[Nx+16*25+n]*Bx->data[Nx+16*10+n] + Ax->data[Nx+16*26+n]*Bx->data[Nx+16*16+n] + Ax->data[Nx+16*27+n]*Bx->data[Nx+16*22+n] + Ax->data[Nx+16*28+n]*Bx->data[Nx+16*28+n] + Ax->data[Nx+16*29+n]*Bx->data[Nx+16*34+n];
      Cx->data[Nx+16*29+n] = Ax->data[Nx+16*24+n]*Bx->data[Nx+16* 5+n] + Ax->data[Nx+16*25+n]*Bx->data[Nx+16*11+n] + Ax->data[Nx+16*26+n]*Bx->data[Nx+16*17+n] + Ax->data[Nx+16*27+n]*Bx->data[Nx+16*23+n] + Ax->data[Nx+16*28+n]*Bx->data[Nx+16*29+n] + Ax->data[Nx+16*29+n]*Bx->data[Nx+16*35+n];
      Cx->data[Nx+16*30+n] = Ax->data[Nx+16*30+n]*Bx->data[Nx+16* 0+n] + Ax->data[Nx+16*31+n]*Bx->data[Nx+16* 6+n] + Ax->data[Nx+16*32+n]*Bx->data[Nx+16*12+n] + Ax->data[Nx+16*33+n]*Bx->data[Nx+16*18+n] + Ax->data[Nx+16*34+n]*Bx->data[Nx+16*24+n] + Ax->data[Nx+16*35+n]*Bx->data[Nx+16*30+n];
      Cx->data[Nx+16*31+n] = Ax->data[Nx+16*30+n]*Bx->data[Nx+16* 1+n] + Ax->data[Nx+16*31+n]*Bx->data[Nx+16* 7+n] + Ax->data[Nx+16*32+n]*Bx->data[Nx+16*13+n] + Ax->data[Nx+16*33+n]*Bx->data[Nx+16*19+n] + Ax->data[Nx+16*34+n]*Bx->data[Nx+16*25+n] + Ax->data[Nx+16*35+n]*Bx->data[Nx+16*31+n];
      Cx->data[Nx+16*32+n] = Ax->data[Nx+16*30+n]*Bx->data[Nx+16* 2+n] + Ax->data[Nx+16*31+n]*Bx->data[Nx+16* 8+n] + Ax->data[Nx+16*32+n]*Bx->data[Nx+16*14+n] + Ax->data[Nx+16*33+n]*Bx->data[Nx+16*20+n] + Ax->data[Nx+16*34+n]*Bx->data[Nx+16*26+n] + Ax->data[Nx+16*35+n]*Bx->data[Nx+16*32+n];
      Cx->data[Nx+16*33+n] = Ax->data[Nx+16*30+n]*Bx->data[Nx+16* 3+n] + Ax->data[Nx+16*31+n]*Bx->data[Nx+16* 9+n] + Ax->data[Nx+16*32+n]*Bx->data[Nx+16*15+n] + Ax->data[Nx+16*33+n]*Bx->data[Nx+16*21+n] + Ax->data[Nx+16*34+n]*Bx->data[Nx+16*27+n] + Ax->data[Nx+16*35+n]*Bx->data[Nx+16*33+n];
      Cx->data[Nx+16*34+n] = Ax->data[Nx+16*30+n]*Bx->data[Nx+16* 4+n] + Ax->data[Nx+16*31+n]*Bx->data[Nx+16*10+n] + Ax->data[Nx+16*32+n]*Bx->data[Nx+16*16+n] + Ax->data[Nx+16*33+n]*Bx->data[Nx+16*22+n] + Ax->data[Nx+16*34+n]*Bx->data[Nx+16*28+n] + Ax->data[Nx+16*35+n]*Bx->data[Nx+16*34+n];
      Cx->data[Nx+16*35+n] = Ax->data[Nx+16*30+n]*Bx->data[Nx+16* 5+n] + Ax->data[Nx+16*31+n]*Bx->data[Nx+16*11+n] + Ax->data[Nx+16*32+n]*Bx->data[Nx+16*17+n] + Ax->data[Nx+16*33+n]*Bx->data[Nx+16*23+n] + Ax->data[Nx+16*34+n]*Bx->data[Nx+16*29+n] + Ax->data[Nx+16*35+n]*Bx->data[Nx+16*35+n];
    }
  }
  // for (size_t x = 0; x < NN/16; ++x) {
  //   const size_t Nx = x*16*36;
  //   for (size_t i = 0; i < 6; ++i) {
  //     for (size_t j = 0; j < 6; ++j) {
  //       for (size_t k = 0; k < 6; ++k) {
  //         for (size_t n = 0; n < 16; ++n) {
  //           Cx->data[ Nx + (i*6 + j)*16 + n ] += Ax->data[ Nx + (i*6 + k)*16 + n ] * Bx->data[ Nx + (k*6 + j)*16 + n];
  //         }
  //       }
  //     }
  //   }
  // }
  const clock_t end = clock();

  // std::cout << "Ax=" << std::endl
  //        << Ax->data[16*(0*6+0)] << " " << Ax->data[16*(0*6+1)] << " " << Ax->data[16*(0*6+2)] << " " << Ax->data[16*(0*6+3)] << " " << Ax->data[16*(0*6+4)] << " " << Ax->data[16*(0*6+5)] << std::endl
  //        << Ax->data[16*(1*6+0)] << " " << Ax->data[16*(1*6+1)] << " " << Ax->data[16*(1*6+2)] << " " << Ax->data[16*(1*6+3)] << " " << Ax->data[16*(1*6+4)] << " " << Ax->data[16*(1*6+5)] << std::endl
  //        << Ax->data[16*(2*6+0)] << " " << Ax->data[16*(2*6+1)] << " " << Ax->data[16*(2*6+2)] << " " << Ax->data[16*(2*6+3)] << " " << Ax->data[16*(2*6+4)] << " " << Ax->data[16*(2*6+5)] << std::endl
  //        << Ax->data[16*(3*6+0)] << " " << Ax->data[16*(3*6+1)] << " " << Ax->data[16*(3*6+2)] << " " << Ax->data[16*(3*6+3)] << " " << Ax->data[16*(3*6+4)] << " " << Ax->data[16*(3*6+5)] << std::endl
  //        << Ax->data[16*(4*6+0)] << " " << Ax->data[16*(4*6+1)] << " " << Ax->data[16*(4*6+2)] << " " << Ax->data[16*(4*6+3)] << " " << Ax->data[16*(4*6+4)] << " " << Ax->data[16*(4*6+5)] << std::endl
  //        << Ax->data[16*(5*6+0)] << " " << Ax->data[16*(5*6+1)] << " " << Ax->data[16*(5*6+2)] << " " << Ax->data[16*(5*6+3)] << " " << Ax->data[16*(5*6+4)] << " " << Ax->data[16*(5*6+5)] << std::endl;
  // std::cout << "Bx=" << std::endl
  //        << Bx->data[16*(0*6+0)] << " " << Bx->data[16*(0*6+1)] << " " << Bx->data[16*(0*6+2)] << " " << Bx->data[16*(0*6+3)] << " " << Bx->data[16*(0*6+4)] << " " << Bx->data[16*(0*6+5)] << std::endl
  //        << Bx->data[16*(1*6+0)] << " " << Bx->data[16*(1*6+1)] << " " << Bx->data[16*(1*6+2)] << " " << Bx->data[16*(1*6+3)] << " " << Bx->data[16*(1*6+4)] << " " << Bx->data[16*(1*6+5)] << std::endl
  //        << Bx->data[16*(2*6+0)] << " " << Bx->data[16*(2*6+1)] << " " << Bx->data[16*(2*6+2)] << " " << Bx->data[16*(2*6+3)] << " " << Bx->data[16*(2*6+4)] << " " << Bx->data[16*(2*6+5)] << std::endl
  //        << Bx->data[16*(3*6+0)] << " " << Bx->data[16*(3*6+1)] << " " << Bx->data[16*(3*6+2)] << " " << Bx->data[16*(3*6+3)] << " " << Bx->data[16*(3*6+4)] << " " << Bx->data[16*(3*6+5)] << std::endl
  //        << Bx->data[16*(4*6+0)] << " " << Bx->data[16*(4*6+1)] << " " << Bx->data[16*(4*6+2)] << " " << Bx->data[16*(4*6+3)] << " " << Bx->data[16*(4*6+4)] << " " << Bx->data[16*(4*6+5)] << std::endl
  //        << Bx->data[16*(5*6+0)] << " " << Bx->data[16*(5*6+1)] << " " << Bx->data[16*(5*6+2)] << " " << Bx->data[16*(5*6+3)] << " " << Bx->data[16*(5*6+4)] << " " << Bx->data[16*(5*6+5)] << std::endl;
  std::cout << "Cx=" << std::endl
            << Cx->data[16*(0*6+0)] << " " << Cx->data[16*(0*6+1)] << " " << Cx->data[16*(0*6+2)] << " " << Cx->data[16*(0*6+3)] << " " << Cx->data[16*(0*6+4)] << " " << Cx->data[16*(0*6+5)] << std::endl
            << Cx->data[16*(1*6+0)] << " " << Cx->data[16*(1*6+1)] << " " << Cx->data[16*(1*6+2)] << " " << Cx->data[16*(1*6+3)] << " " << Cx->data[16*(1*6+4)] << " " << Cx->data[16*(1*6+5)] << std::endl
            << Cx->data[16*(2*6+0)] << " " << Cx->data[16*(2*6+1)] << " " << Cx->data[16*(2*6+2)] << " " << Cx->data[16*(2*6+3)] << " " << Cx->data[16*(2*6+4)] << " " << Cx->data[16*(2*6+5)] << std::endl
            << Cx->data[16*(3*6+0)] << " " << Cx->data[16*(3*6+1)] << " " << Cx->data[16*(3*6+2)] << " " << Cx->data[16*(3*6+3)] << " " << Cx->data[16*(3*6+4)] << " " << Cx->data[16*(3*6+5)] << std::endl
            << Cx->data[16*(4*6+0)] << " " << Cx->data[16*(4*6+1)] << " " << Cx->data[16*(4*6+2)] << " " << Cx->data[16*(4*6+3)] << " " << Cx->data[16*(4*6+4)] << " " << Cx->data[16*(4*6+5)] << std::endl
            << Cx->data[16*(5*6+0)] << " " << Cx->data[16*(5*6+1)] << " " << Cx->data[16*(5*6+2)] << " " << Cx->data[16*(5*6+3)] << " " << Cx->data[16*(5*6+4)] << " " << Cx->data[16*(5*6+5)] << std::endl;
  float time = float(end-begin)/CLOCKS_PER_SEC;
  std::cout << "plain -- time for NN=" << NN << " multiplications is " << time << " s, i.e. per track [s]=" << time/float(NN) << std::endl;
  delete Ax, Bx, Cx;
}

int main(int argc, char* argv[])
{


  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0.0, 100.0);

  std::array<std::array<float, NN>, 36> input;
  for (size_t i=0;i<36;++i) {
    for (size_t j=0;j<NN;++j){
      input[i][j] = (float)dis(gen);
    }
  }
  
  std::cout << "done preparing input" << std::endl;

  test_plain(input);
  test_plain(input); // twice to warm up
  test_plain(input);

  return 0;
}
