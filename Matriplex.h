#include <iostream>
#include <omp.h>

void plainArray_matrix_mult66(const float* Ax, const float* Bx, float* Cx, int NN) {
  for (size_t x = 0; x < NN; ++x) {
    const size_t Nx = x*36;
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
	Cx[ Nx + (i*6 + j) ] = 0.;
	for (size_t k = 0; k < 6; ++k) {
	  Cx[ Nx + (i*6 + j) ] += Ax[ Nx + (i*6 + k) ] * Bx[ Nx + (k*6 + j) ];
	}
      }
    }
  }
}

/*This one is real real slow (7000 s vs 30 s for non parallel version)
void plainArray_matrix_mult66_par(const float* Ax, const float* Bx, float* Cx, int NN) {
  // #pragma omp simd
  //#pragma omp parallel for
  //{
  for (size_t x = 0; x < NN; ++x) {
    const size_t Nx = x*36;
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        Cx[ Nx + (i*6 + j) ] = 0.;
        float temp_sum = 0.0;
#pragma omp parallel
	{
#pragma omp for reduction(+:temp_sum)
	//	{
        for (size_t k = 0; k < 6; ++k) {
          temp_sum += Ax[ Nx + (i*6 + k) ] * Bx[ Nx + (k*6 + j) ];
        }
	}//end of parallel region
	Cx[ Nx + (i*6 + j) ] = temp_sum;
      }
    }
  }
  }*/

void plainArray_matrix_mult66_par(const float* Ax, const float* Bx, float* Cx, int NN) {
#pragma omp parallel for 
  for (size_t x = 0; x < NN; ++x) {
    size_t Nx = x*36;
    //    printf("threadId = %d \n", omp_get_thread_num());
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
	for (size_t k = 0; k < 6; ++k) {
	  Cx[ Nx + (i*6 + j) ] += Ax[ Nx + (i*6 + k) ] * Bx[ Nx + (k*6 + j) ];
	}      
      }
    }
  }
}

void plainArray_element_mult66(const float* Ax, const float* Bx, float* Cx, int NN) {
  for (size_t x = 0; x < NN; ++x) {
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
	Cx[ x + (i*6 + j)*NN ] = 0.;
        for (size_t k = 0; k < 6; ++k) {
          Cx[ x + (i*6 + j)*NN ] += Ax[ x + (i*6 + k)*NN ] * Bx[ x + (k*6 + j)*NN ];
        }
      }
    }
  }
}

void plainArray_el16mx_mult66(const float* Ax, const float* Bx, float* Cx, int NN) {
  for (size_t x = 0; x < NN/16; ++x) {
    const size_t Nx = x*16*36;
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
	//
	for (size_t n = 0; n < 16; ++n) {
	  Cx[ Nx + (i*6 + j)*16 + n ] = 0.;
	}
	//
	for (size_t k = 0; k < 6; ++k) {
#pragma omp simd
	  for (size_t n = 0; n < 16; ++n) {
	    Cx[ Nx + (i*6 + j)*16 + n ] += Ax[ Nx + (i*6 + k)*16 + n ] * Bx[ Nx + (k*6 + j)*16 + n];
	  }
	}
      }
    }
  }
}

void plainArray_el16mx_mult66_v1(const float* Ax, const float* Bx, float* Cx, int NN) {
  for (size_t x = 0; x < NN/16; ++x) {
    const size_t Nx = x*16*36;
#pragma omp simd
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
}

