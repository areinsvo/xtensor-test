#include <iostream>
#include "xtensor/xarray.hpp"
#include <caliper/cali.h>
#include <stdlib.h>
#include <stdio.h>

class MatriplXT66
{
  //
public:
  //
  xt::xarray<float>& operator[] (size_t k) { return m_[k]; }
  const xt::xarray<float>& operator[] (size_t k) const { return m_[k]; }
  //
  xt::xarray<float>& operator() (size_t i, size_t j) { return m_[i * 6 + j]; }
  const xt::xarray<float>& operator() (size_t i, size_t j) const { return m_[i * 6 + j]; }
  //
  float operator() (size_t i, size_t j, size_t n) const { return m_[i * 6 + j](n); }
  //
  void print(std::ostream& os, size_t n)
  {
    for (size_t i=0; i<6; i++) {
      for (size_t j=0; j<6; j++) {
        os << m_[i * 6 + j](n) << "\t";
      }
      os << std::endl;
    }
  }
  //
private:
  //
  std::array<xt::xarray<float>, 36> m_;
};

class MatriplXT66_v1
{
  //
public:
  MatriplXT66_v1(size_t NN) {
    std::vector<size_t> shape = {NN, 36};
    m_ = xt::xarray<float, xt::layout_type::column_major>(shape);
  }
  MatriplXT66_v1(xt::xarray<float, xt::layout_type::column_major>&& m) noexcept
    : m_(std::move(m)) {}
  //
  auto operator[] (size_t k) { return xt::view(m_,xt::all(),k); }
  const auto operator[] (size_t k) const { return xt::view(m_,xt::all(),k); }
  //
  auto operator() (size_t i, size_t j) { return xt::view(m_,xt::all(),i * 6 + j); }
  const auto operator() (size_t i, size_t j) const { return xt::view(m_,xt::all(),i * 6 + j); }
  float operator() (size_t i, size_t j, size_t n) const { return m_(n,i * 6 + j); }
  void print(std::ostream& os, size_t n)
  {
    for (size_t i=0; i<6; i++) {
      for (size_t j=0; j<6; j++) {
        os << m_(n,i * 6 + j) << "\t";
      }
      os << std::endl;
    }
  }
  //
private:
  //
  xt::xarray<float, xt::layout_type::column_major> m_;
};

class MatriplXT66_v2
{
  //
public:
  MatriplXT66_v2(size_t NN)
    : m_({NN, 36}) {}
  MatriplXT66_v2(xt::xtensor<float, 2, xt::layout_type::column_major>&& m) noexcept
    : m_(std::move(m)) {}
  //
  auto operator[] (size_t k) { return xt::view(m_,xt::all(),k); }
  const auto operator[] (size_t k) const { return xt::view(m_,xt::all(),k); }
  //
  auto operator() (size_t i, size_t j) { return xt::view(m_,xt::all(),i * 6 + j); }
  const auto operator() (size_t i, size_t j) const { return xt::view(m_,xt::all(),i * 6 + j); }
  //
  float operator() (size_t i, size_t j, size_t n) const { return m_(n,i * 6 + j); }
  void print(std::ostream& os, size_t n)
  {
    for (size_t i=0; i<6; i++) {
      for (size_t j=0; j<6; j++) {
        os << m_(n,i * 6 + j) << "\t";
      }
      os << std::endl;
    }
  }
  //
private:
  //
  xt::xtensor<float, 2, xt::layout_type::column_major> m_;
};

void MultiplyXT66(const MatriplXT66& A,
                  const MatriplXT66& B,
                        MatriplXT66& C)
{
  #ifdef USE_CALI
  CALI_CXX_MARK_FUNCTION;
  #endif
  C[ 0] = A[ 0]*B[ 0] + A[ 1]*B[ 6] + A[ 2]*B[12] + A[ 3]*B[18] + A[ 4]*B[24] + A[ 5]*B[30];
  C[ 1] = A[ 0]*B[ 1] + A[ 1]*B[ 7] + A[ 2]*B[13] + A[ 3]*B[19] + A[ 4]*B[25] + A[ 5]*B[31];
  C[ 2] = A[ 0]*B[ 2] + A[ 1]*B[ 8] + A[ 2]*B[14] + A[ 3]*B[20] + A[ 4]*B[26] + A[ 5]*B[32];
  C[ 3] = A[ 0]*B[ 3] + A[ 1]*B[ 9] + A[ 2]*B[15] + A[ 3]*B[21] + A[ 4]*B[27] + A[ 5]*B[33];
  C[ 4] = A[ 0]*B[ 4] + A[ 1]*B[10] + A[ 2]*B[16] + A[ 3]*B[22] + A[ 4]*B[28] + A[ 5]*B[34];
  C[ 5] = A[ 0]*B[ 5] + A[ 1]*B[11] + A[ 2]*B[17] + A[ 3]*B[23] + A[ 4]*B[29] + A[ 5]*B[35];
  C[ 6] = A[ 6]*B[ 0] + A[ 7]*B[ 6] + A[ 8]*B[12] + A[ 9]*B[18] + A[10]*B[24] + A[11]*B[30];
  C[ 7] = A[ 6]*B[ 1] + A[ 7]*B[ 7] + A[ 8]*B[13] + A[ 9]*B[19] + A[10]*B[25] + A[11]*B[31];
  C[ 8] = A[ 6]*B[ 2] + A[ 7]*B[ 8] + A[ 8]*B[14] + A[ 9]*B[20] + A[10]*B[26] + A[11]*B[32];
  C[ 9] = A[ 6]*B[ 3] + A[ 7]*B[ 9] + A[ 8]*B[15] + A[ 9]*B[21] + A[10]*B[27] + A[11]*B[33];
  C[10] = A[ 6]*B[ 4] + A[ 7]*B[10] + A[ 8]*B[16] + A[ 9]*B[22] + A[10]*B[28] + A[11]*B[34];
  C[11] = A[ 6]*B[ 5] + A[ 7]*B[11] + A[ 8]*B[17] + A[ 9]*B[23] + A[10]*B[29] + A[11]*B[35];
  C[12] = A[12]*B[ 0] + A[13]*B[ 6] + A[14]*B[12] + A[15]*B[18] + A[16]*B[24] + A[17]*B[30];
  C[13] = A[12]*B[ 1] + A[13]*B[ 7] + A[14]*B[13] + A[15]*B[19] + A[16]*B[25] + A[17]*B[31];
  C[14] = A[12]*B[ 2] + A[13]*B[ 8] + A[14]*B[14] + A[15]*B[20] + A[16]*B[26] + A[17]*B[32];
  C[15] = A[12]*B[ 3] + A[13]*B[ 9] + A[14]*B[15] + A[15]*B[21] + A[16]*B[27] + A[17]*B[33];
  C[16] = A[12]*B[ 4] + A[13]*B[10] + A[14]*B[16] + A[15]*B[22] + A[16]*B[28] + A[17]*B[34];
  C[17] = A[12]*B[ 5] + A[13]*B[11] + A[14]*B[17] + A[15]*B[23] + A[16]*B[29] + A[17]*B[35];
  C[18] = A[18]*B[ 0] + A[19]*B[ 6] + A[20]*B[12] + A[21]*B[18] + A[22]*B[24] + A[23]*B[30];
  C[19] = A[18]*B[ 1] + A[19]*B[ 7] + A[20]*B[13] + A[21]*B[19] + A[22]*B[25] + A[23]*B[31];
  C[20] = A[18]*B[ 2] + A[19]*B[ 8] + A[20]*B[14] + A[21]*B[20] + A[22]*B[26] + A[23]*B[32];
  C[21] = A[18]*B[ 3] + A[19]*B[ 9] + A[20]*B[15] + A[21]*B[21] + A[22]*B[27] + A[23]*B[33];
  C[22] = A[18]*B[ 4] + A[19]*B[10] + A[20]*B[16] + A[21]*B[22] + A[22]*B[28] + A[23]*B[34];
  C[23] = A[18]*B[ 5] + A[19]*B[11] + A[20]*B[17] + A[21]*B[23] + A[22]*B[29] + A[23]*B[35];
  C[24] = A[24]*B[ 0] + A[25]*B[ 6] + A[26]*B[12] + A[27]*B[18] + A[28]*B[24] + A[29]*B[30];
  C[25] = A[24]*B[ 1] + A[25]*B[ 7] + A[26]*B[13] + A[27]*B[19] + A[28]*B[25] + A[29]*B[31];
  C[26] = A[24]*B[ 2] + A[25]*B[ 8] + A[26]*B[14] + A[27]*B[20] + A[28]*B[26] + A[29]*B[32];
  C[27] = A[24]*B[ 3] + A[25]*B[ 9] + A[26]*B[15] + A[27]*B[21] + A[28]*B[27] + A[29]*B[33];
  C[28] = A[24]*B[ 4] + A[25]*B[10] + A[26]*B[16] + A[27]*B[22] + A[28]*B[28] + A[29]*B[34];
  C[29] = A[24]*B[ 5] + A[25]*B[11] + A[26]*B[17] + A[27]*B[23] + A[28]*B[29] + A[29]*B[35];
  C[30] = A[30]*B[ 0] + A[31]*B[ 6] + A[32]*B[12] + A[33]*B[18] + A[34]*B[24] + A[35]*B[30];
  C[31] = A[30]*B[ 1] + A[31]*B[ 7] + A[32]*B[13] + A[33]*B[19] + A[34]*B[25] + A[35]*B[31];
  C[32] = A[30]*B[ 2] + A[31]*B[ 8] + A[32]*B[14] + A[33]*B[20] + A[34]*B[26] + A[35]*B[32];
  C[33] = A[30]*B[ 3] + A[31]*B[ 9] + A[32]*B[15] + A[33]*B[21] + A[34]*B[27] + A[35]*B[33];
  C[34] = A[30]*B[ 4] + A[31]*B[10] + A[32]*B[16] + A[33]*B[22] + A[34]*B[28] + A[35]*B[34];
  C[35] = A[30]*B[ 5] + A[31]*B[11] + A[32]*B[17] + A[33]*B[23] + A[34]*B[29] + A[35]*B[35];
}

void MultiplyXT66(const MatriplXT66_v1& A,
                  const MatriplXT66_v1& B,
                        MatriplXT66_v1& C)
{
  #ifdef USE_CALI
  CALI_CXX_MARK_FUNCTION;
  #endif
  C[ 0] = A[ 0]*B[ 0] + A[ 1]*B[ 6] + A[ 2]*B[12] + A[ 3]*B[18] + A[ 4]*B[24] + A[ 5]*B[30];
  C[ 1] = A[ 0]*B[ 1] + A[ 1]*B[ 7] + A[ 2]*B[13] + A[ 3]*B[19] + A[ 4]*B[25] + A[ 5]*B[31];
  C[ 2] = A[ 0]*B[ 2] + A[ 1]*B[ 8] + A[ 2]*B[14] + A[ 3]*B[20] + A[ 4]*B[26] + A[ 5]*B[32];
  C[ 3] = A[ 0]*B[ 3] + A[ 1]*B[ 9] + A[ 2]*B[15] + A[ 3]*B[21] + A[ 4]*B[27] + A[ 5]*B[33];
  C[ 4] = A[ 0]*B[ 4] + A[ 1]*B[10] + A[ 2]*B[16] + A[ 3]*B[22] + A[ 4]*B[28] + A[ 5]*B[34];
  C[ 5] = A[ 0]*B[ 5] + A[ 1]*B[11] + A[ 2]*B[17] + A[ 3]*B[23] + A[ 4]*B[29] + A[ 5]*B[35];
  C[ 6] = A[ 6]*B[ 0] + A[ 7]*B[ 6] + A[ 8]*B[12] + A[ 9]*B[18] + A[10]*B[24] + A[11]*B[30];
  C[ 7] = A[ 6]*B[ 1] + A[ 7]*B[ 7] + A[ 8]*B[13] + A[ 9]*B[19] + A[10]*B[25] + A[11]*B[31];
  C[ 8] = A[ 6]*B[ 2] + A[ 7]*B[ 8] + A[ 8]*B[14] + A[ 9]*B[20] + A[10]*B[26] + A[11]*B[32];
  C[ 9] = A[ 6]*B[ 3] + A[ 7]*B[ 9] + A[ 8]*B[15] + A[ 9]*B[21] + A[10]*B[27] + A[11]*B[33];
  C[10] = A[ 6]*B[ 4] + A[ 7]*B[10] + A[ 8]*B[16] + A[ 9]*B[22] + A[10]*B[28] + A[11]*B[34];
  C[11] = A[ 6]*B[ 5] + A[ 7]*B[11] + A[ 8]*B[17] + A[ 9]*B[23] + A[10]*B[29] + A[11]*B[35];
  C[12] = A[12]*B[ 0] + A[13]*B[ 6] + A[14]*B[12] + A[15]*B[18] + A[16]*B[24] + A[17]*B[30];
  C[13] = A[12]*B[ 1] + A[13]*B[ 7] + A[14]*B[13] + A[15]*B[19] + A[16]*B[25] + A[17]*B[31];
  C[14] = A[12]*B[ 2] + A[13]*B[ 8] + A[14]*B[14] + A[15]*B[20] + A[16]*B[26] + A[17]*B[32];
  C[15] = A[12]*B[ 3] + A[13]*B[ 9] + A[14]*B[15] + A[15]*B[21] + A[16]*B[27] + A[17]*B[33];
  C[16] = A[12]*B[ 4] + A[13]*B[10] + A[14]*B[16] + A[15]*B[22] + A[16]*B[28] + A[17]*B[34];
  C[17] = A[12]*B[ 5] + A[13]*B[11] + A[14]*B[17] + A[15]*B[23] + A[16]*B[29] + A[17]*B[35];
  C[18] = A[18]*B[ 0] + A[19]*B[ 6] + A[20]*B[12] + A[21]*B[18] + A[22]*B[24] + A[23]*B[30];
  C[19] = A[18]*B[ 1] + A[19]*B[ 7] + A[20]*B[13] + A[21]*B[19] + A[22]*B[25] + A[23]*B[31];
  C[20] = A[18]*B[ 2] + A[19]*B[ 8] + A[20]*B[14] + A[21]*B[20] + A[22]*B[26] + A[23]*B[32];
  C[21] = A[18]*B[ 3] + A[19]*B[ 9] + A[20]*B[15] + A[21]*B[21] + A[22]*B[27] + A[23]*B[33];
  C[22] = A[18]*B[ 4] + A[19]*B[10] + A[20]*B[16] + A[21]*B[22] + A[22]*B[28] + A[23]*B[34];
  C[23] = A[18]*B[ 5] + A[19]*B[11] + A[20]*B[17] + A[21]*B[23] + A[22]*B[29] + A[23]*B[35];
  C[24] = A[24]*B[ 0] + A[25]*B[ 6] + A[26]*B[12] + A[27]*B[18] + A[28]*B[24] + A[29]*B[30];
  C[25] = A[24]*B[ 1] + A[25]*B[ 7] + A[26]*B[13] + A[27]*B[19] + A[28]*B[25] + A[29]*B[31];
  C[26] = A[24]*B[ 2] + A[25]*B[ 8] + A[26]*B[14] + A[27]*B[20] + A[28]*B[26] + A[29]*B[32];
  C[27] = A[24]*B[ 3] + A[25]*B[ 9] + A[26]*B[15] + A[27]*B[21] + A[28]*B[27] + A[29]*B[33];
  C[28] = A[24]*B[ 4] + A[25]*B[10] + A[26]*B[16] + A[27]*B[22] + A[28]*B[28] + A[29]*B[34];
  C[29] = A[24]*B[ 5] + A[25]*B[11] + A[26]*B[17] + A[27]*B[23] + A[28]*B[29] + A[29]*B[35];
  C[30] = A[30]*B[ 0] + A[31]*B[ 6] + A[32]*B[12] + A[33]*B[18] + A[34]*B[24] + A[35]*B[30];
  C[31] = A[30]*B[ 1] + A[31]*B[ 7] + A[32]*B[13] + A[33]*B[19] + A[34]*B[25] + A[35]*B[31];
  C[32] = A[30]*B[ 2] + A[31]*B[ 8] + A[32]*B[14] + A[33]*B[20] + A[34]*B[26] + A[35]*B[32];
  C[33] = A[30]*B[ 3] + A[31]*B[ 9] + A[32]*B[15] + A[33]*B[21] + A[34]*B[27] + A[35]*B[33];
  C[34] = A[30]*B[ 4] + A[31]*B[10] + A[32]*B[16] + A[33]*B[22] + A[34]*B[28] + A[35]*B[34];
  C[35] = A[30]*B[ 5] + A[31]*B[11] + A[32]*B[17] + A[33]*B[23] + A[34]*B[29] + A[35]*B[35];
}

void MultiplyXT66(const MatriplXT66_v2& A,
                  const MatriplXT66_v2& B,
                        MatriplXT66_v2& C)
{
  #ifdef USE_CALI
  CALI_CXX_MARK_FUNCTION;
  #endif
  C[ 0] = A[ 0]*B[ 0] + A[ 1]*B[ 6] + A[ 2]*B[12] + A[ 3]*B[18] + A[ 4]*B[24] + A[ 5]*B[30];
  C[ 1] = A[ 0]*B[ 1] + A[ 1]*B[ 7] + A[ 2]*B[13] + A[ 3]*B[19] + A[ 4]*B[25] + A[ 5]*B[31];
  C[ 2] = A[ 0]*B[ 2] + A[ 1]*B[ 8] + A[ 2]*B[14] + A[ 3]*B[20] + A[ 4]*B[26] + A[ 5]*B[32];
  C[ 3] = A[ 0]*B[ 3] + A[ 1]*B[ 9] + A[ 2]*B[15] + A[ 3]*B[21] + A[ 4]*B[27] + A[ 5]*B[33];
  C[ 4] = A[ 0]*B[ 4] + A[ 1]*B[10] + A[ 2]*B[16] + A[ 3]*B[22] + A[ 4]*B[28] + A[ 5]*B[34];
  C[ 5] = A[ 0]*B[ 5] + A[ 1]*B[11] + A[ 2]*B[17] + A[ 3]*B[23] + A[ 4]*B[29] + A[ 5]*B[35];
  C[ 6] = A[ 6]*B[ 0] + A[ 7]*B[ 6] + A[ 8]*B[12] + A[ 9]*B[18] + A[10]*B[24] + A[11]*B[30];
  C[ 7] = A[ 6]*B[ 1] + A[ 7]*B[ 7] + A[ 8]*B[13] + A[ 9]*B[19] + A[10]*B[25] + A[11]*B[31];
  C[ 8] = A[ 6]*B[ 2] + A[ 7]*B[ 8] + A[ 8]*B[14] + A[ 9]*B[20] + A[10]*B[26] + A[11]*B[32];
  C[ 9] = A[ 6]*B[ 3] + A[ 7]*B[ 9] + A[ 8]*B[15] + A[ 9]*B[21] + A[10]*B[27] + A[11]*B[33];
  C[10] = A[ 6]*B[ 4] + A[ 7]*B[10] + A[ 8]*B[16] + A[ 9]*B[22] + A[10]*B[28] + A[11]*B[34];
  C[11] = A[ 6]*B[ 5] + A[ 7]*B[11] + A[ 8]*B[17] + A[ 9]*B[23] + A[10]*B[29] + A[11]*B[35];
  C[12] = A[12]*B[ 0] + A[13]*B[ 6] + A[14]*B[12] + A[15]*B[18] + A[16]*B[24] + A[17]*B[30];
  C[13] = A[12]*B[ 1] + A[13]*B[ 7] + A[14]*B[13] + A[15]*B[19] + A[16]*B[25] + A[17]*B[31];
  C[14] = A[12]*B[ 2] + A[13]*B[ 8] + A[14]*B[14] + A[15]*B[20] + A[16]*B[26] + A[17]*B[32];
  C[15] = A[12]*B[ 3] + A[13]*B[ 9] + A[14]*B[15] + A[15]*B[21] + A[16]*B[27] + A[17]*B[33];
  C[16] = A[12]*B[ 4] + A[13]*B[10] + A[14]*B[16] + A[15]*B[22] + A[16]*B[28] + A[17]*B[34];
  C[17] = A[12]*B[ 5] + A[13]*B[11] + A[14]*B[17] + A[15]*B[23] + A[16]*B[29] + A[17]*B[35];
  C[18] = A[18]*B[ 0] + A[19]*B[ 6] + A[20]*B[12] + A[21]*B[18] + A[22]*B[24] + A[23]*B[30];
  C[19] = A[18]*B[ 1] + A[19]*B[ 7] + A[20]*B[13] + A[21]*B[19] + A[22]*B[25] + A[23]*B[31];
  C[20] = A[18]*B[ 2] + A[19]*B[ 8] + A[20]*B[14] + A[21]*B[20] + A[22]*B[26] + A[23]*B[32];
  C[21] = A[18]*B[ 3] + A[19]*B[ 9] + A[20]*B[15] + A[21]*B[21] + A[22]*B[27] + A[23]*B[33];
  C[22] = A[18]*B[ 4] + A[19]*B[10] + A[20]*B[16] + A[21]*B[22] + A[22]*B[28] + A[23]*B[34];
  C[23] = A[18]*B[ 5] + A[19]*B[11] + A[20]*B[17] + A[21]*B[23] + A[22]*B[29] + A[23]*B[35];
  C[24] = A[24]*B[ 0] + A[25]*B[ 6] + A[26]*B[12] + A[27]*B[18] + A[28]*B[24] + A[29]*B[30];
  C[25] = A[24]*B[ 1] + A[25]*B[ 7] + A[26]*B[13] + A[27]*B[19] + A[28]*B[25] + A[29]*B[31];
  C[26] = A[24]*B[ 2] + A[25]*B[ 8] + A[26]*B[14] + A[27]*B[20] + A[28]*B[26] + A[29]*B[32];
  C[27] = A[24]*B[ 3] + A[25]*B[ 9] + A[26]*B[15] + A[27]*B[21] + A[28]*B[27] + A[29]*B[33];
  C[28] = A[24]*B[ 4] + A[25]*B[10] + A[26]*B[16] + A[27]*B[22] + A[28]*B[28] + A[29]*B[34];
  C[29] = A[24]*B[ 5] + A[25]*B[11] + A[26]*B[17] + A[27]*B[23] + A[28]*B[29] + A[29]*B[35];
  C[30] = A[30]*B[ 0] + A[31]*B[ 6] + A[32]*B[12] + A[33]*B[18] + A[34]*B[24] + A[35]*B[30];
  C[31] = A[30]*B[ 1] + A[31]*B[ 7] + A[32]*B[13] + A[33]*B[19] + A[34]*B[25] + A[35]*B[31];
  C[32] = A[30]*B[ 2] + A[31]*B[ 8] + A[32]*B[14] + A[33]*B[20] + A[34]*B[26] + A[35]*B[32];
  C[33] = A[30]*B[ 3] + A[31]*B[ 9] + A[32]*B[15] + A[33]*B[21] + A[34]*B[27] + A[35]*B[33];
  C[34] = A[30]*B[ 4] + A[31]*B[10] + A[32]*B[16] + A[33]*B[22] + A[34]*B[28] + A[35]*B[34];
  C[35] = A[30]*B[ 5] + A[31]*B[11] + A[32]*B[17] + A[33]*B[23] + A[34]*B[29] + A[35]*B[35];
}

void MultiplyXT66LoopTile(const MatriplXT66& A,
                          const MatriplXT66& B,
                          MatriplXT66& C)
{
  #ifdef USE_CALI
  CALI_CXX_MARK_FUNCTION;
  #endif
  #define TILE 2
  /* Loop over all the tiles, stride by tile size */
  for ( int it=0; it<6; it+=TILE ) {
    for ( int jt=0; jt<6; jt+=TILE ) {
      for ( int kt=0; kt<6; kt+=TILE ) {
/* Regular multiply inside the tiles */
        for (size_t i=it; i<it+TILE; ++i) {
          for (size_t j=jt; j<jt+TILE; ++j) {
            //C(i,j) = 0;
            for (size_t k=kt; k<kt+TILE; ++k) {
              C(i,j) += A(i,k)*B(k,j);
            }
          }
        }
      }
    }
  }
}

void MultiplyXT66Loop(const MatriplXT66& A,
                      const MatriplXT66& B,
                      MatriplXT66& C)
{
  #ifdef USE_CALI
  CALI_CXX_MARK_FUNCTION;
  #endif
  for (size_t i=0; i<6; ++i) {
    for (size_t j=0; j<6; ++j) {
      C(i,j) = 0;
      for (size_t k=0; k<6; ++k) {
        C(i,j) += A(i,k)*B(k,j);
      }
    }
  }
}

void MultiplyXT66Loop(const MatriplXT66_v1& A,
                      const MatriplXT66_v1& B,
                      MatriplXT66_v1& C)
{
  #ifdef USE_CALI
  CALI_CXX_MARK_FUNCTION;
  #endif
  for (size_t i=0; i<6; ++i) {
    for (size_t j=0; j<6; ++j) {
      C(i,j) = 0;
      for (size_t k=0; k<6; ++k) {
        C(i,j) += A(i,k)*B(k,j);
      }
    }
  }
}

void MultiplyXT66Loop(const MatriplXT66_v2& A,
                      const MatriplXT66_v2& B,
                      MatriplXT66_v2& C)
{
  #ifdef USE_CALI
  CALI_CXX_MARK_FUNCTION;
  #endif
  for (size_t i=0; i<6; ++i) {
    for (size_t j=0; j<6; ++j) {
      C(i,j) = 0;
      for (size_t k=0; k<6; ++k) {
        C(i,j) += A(i,k)*B(k,j);
      }
    }
  }
}

void MultiplyXT66Stack(const MatriplXT66_v1& A,
                       const MatriplXT66_v1& B,
                       MatriplXT66_v1& C)
{
  #ifdef USE_CALI
  CALI_CXX_MARK_FUNCTION;
  #endif
  auto C00 = A[ 0]*B[ 0] + A[ 1]*B[ 6] + A[ 2]*B[12] + A[ 3]*B[18] + A[ 4]*B[24] + A[ 5]*B[30];
  auto C01 = A[ 0]*B[ 1] + A[ 1]*B[ 7] + A[ 2]*B[13] + A[ 3]*B[19] + A[ 4]*B[25] + A[ 5]*B[31];
  auto C02 = A[ 0]*B[ 2] + A[ 1]*B[ 8] + A[ 2]*B[14] + A[ 3]*B[20] + A[ 4]*B[26] + A[ 5]*B[32];
  auto C03 = A[ 0]*B[ 3] + A[ 1]*B[ 9] + A[ 2]*B[15] + A[ 3]*B[21] + A[ 4]*B[27] + A[ 5]*B[33];
  auto C04 = A[ 0]*B[ 4] + A[ 1]*B[10] + A[ 2]*B[16] + A[ 3]*B[22] + A[ 4]*B[28] + A[ 5]*B[34];
  auto C05 = A[ 0]*B[ 5] + A[ 1]*B[11] + A[ 2]*B[17] + A[ 3]*B[23] + A[ 4]*B[29] + A[ 5]*B[35];
  auto C06 = A[ 6]*B[ 0] + A[ 7]*B[ 6] + A[ 8]*B[12] + A[ 9]*B[18] + A[10]*B[24] + A[11]*B[30];
  auto C07 = A[ 6]*B[ 1] + A[ 7]*B[ 7] + A[ 8]*B[13] + A[ 9]*B[19] + A[10]*B[25] + A[11]*B[31];
  auto C08 = A[ 6]*B[ 2] + A[ 7]*B[ 8] + A[ 8]*B[14] + A[ 9]*B[20] + A[10]*B[26] + A[11]*B[32];
  auto C09 = A[ 6]*B[ 3] + A[ 7]*B[ 9] + A[ 8]*B[15] + A[ 9]*B[21] + A[10]*B[27] + A[11]*B[33];
  auto C10 = A[ 6]*B[ 4] + A[ 7]*B[10] + A[ 8]*B[16] + A[ 9]*B[22] + A[10]*B[28] + A[11]*B[34];
  auto C11 = A[ 6]*B[ 5] + A[ 7]*B[11] + A[ 8]*B[17] + A[ 9]*B[23] + A[10]*B[29] + A[11]*B[35];
  auto C12 = A[12]*B[ 0] + A[13]*B[ 6] + A[14]*B[12] + A[15]*B[18] + A[16]*B[24] + A[17]*B[30];
  auto C13 = A[12]*B[ 1] + A[13]*B[ 7] + A[14]*B[13] + A[15]*B[19] + A[16]*B[25] + A[17]*B[31];
  auto C14 = A[12]*B[ 2] + A[13]*B[ 8] + A[14]*B[14] + A[15]*B[20] + A[16]*B[26] + A[17]*B[32];
  auto C15 = A[12]*B[ 3] + A[13]*B[ 9] + A[14]*B[15] + A[15]*B[21] + A[16]*B[27] + A[17]*B[33];
  auto C16 = A[12]*B[ 4] + A[13]*B[10] + A[14]*B[16] + A[15]*B[22] + A[16]*B[28] + A[17]*B[34];
  auto C17 = A[12]*B[ 5] + A[13]*B[11] + A[14]*B[17] + A[15]*B[23] + A[16]*B[29] + A[17]*B[35];
  auto C18 = A[18]*B[ 0] + A[19]*B[ 6] + A[20]*B[12] + A[21]*B[18] + A[22]*B[24] + A[23]*B[30];
  auto C19 = A[18]*B[ 1] + A[19]*B[ 7] + A[20]*B[13] + A[21]*B[19] + A[22]*B[25] + A[23]*B[31];
  auto C20 = A[18]*B[ 2] + A[19]*B[ 8] + A[20]*B[14] + A[21]*B[20] + A[22]*B[26] + A[23]*B[32];
  auto C21 = A[18]*B[ 3] + A[19]*B[ 9] + A[20]*B[15] + A[21]*B[21] + A[22]*B[27] + A[23]*B[33];
  auto C22 = A[18]*B[ 4] + A[19]*B[10] + A[20]*B[16] + A[21]*B[22] + A[22]*B[28] + A[23]*B[34];
  auto C23 = A[18]*B[ 5] + A[19]*B[11] + A[20]*B[17] + A[21]*B[23] + A[22]*B[29] + A[23]*B[35];
  auto C24 = A[24]*B[ 0] + A[25]*B[ 6] + A[26]*B[12] + A[27]*B[18] + A[28]*B[24] + A[29]*B[30];
  auto C25 = A[24]*B[ 1] + A[25]*B[ 7] + A[26]*B[13] + A[27]*B[19] + A[28]*B[25] + A[29]*B[31];
  auto C26 = A[24]*B[ 2] + A[25]*B[ 8] + A[26]*B[14] + A[27]*B[20] + A[28]*B[26] + A[29]*B[32];
  auto C27 = A[24]*B[ 3] + A[25]*B[ 9] + A[26]*B[15] + A[27]*B[21] + A[28]*B[27] + A[29]*B[33];
  auto C28 = A[24]*B[ 4] + A[25]*B[10] + A[26]*B[16] + A[27]*B[22] + A[28]*B[28] + A[29]*B[34];
  auto C29 = A[24]*B[ 5] + A[25]*B[11] + A[26]*B[17] + A[27]*B[23] + A[28]*B[29] + A[29]*B[35];
  auto C30 = A[30]*B[ 0] + A[31]*B[ 6] + A[32]*B[12] + A[33]*B[18] + A[34]*B[24] + A[35]*B[30];
  auto C31 = A[30]*B[ 1] + A[31]*B[ 7] + A[32]*B[13] + A[33]*B[19] + A[34]*B[25] + A[35]*B[31];
  auto C32 = A[30]*B[ 2] + A[31]*B[ 8] + A[32]*B[14] + A[33]*B[20] + A[34]*B[26] + A[35]*B[32];
  auto C33 = A[30]*B[ 3] + A[31]*B[ 9] + A[32]*B[15] + A[33]*B[21] + A[34]*B[27] + A[35]*B[33];
  auto C34 = A[30]*B[ 4] + A[31]*B[10] + A[32]*B[16] + A[33]*B[22] + A[34]*B[28] + A[35]*B[34];
  auto C35 = A[30]*B[ 5] + A[31]*B[11] + A[32]*B[17] + A[33]*B[23] + A[34]*B[29] + A[35]*B[35];
  // trying to force the evaluation all in one place...
  // tried also xt::concatenate instead of xt::stack but the output is not right not sure why
  C = MatriplXT66_v1(xt::stack(xt::xtuple(C00, C01, C02, C03, C04, C05, C06, C07, C08, C09,
					  C10, C11, C12, C13, C14, C15, C16, C17, C18, C19,
					  C20, C21, C22, C23, C24, C25, C26, C27, C28, C29,
					  C30, C31, C32, C33, C34, C35),1));
}

void MultiplyXT66Stack(const MatriplXT66_v2& A,
		       const MatriplXT66_v2& B,
		       MatriplXT66_v2& C)
{
  #ifdef USE_CALI
  CALI_CXX_MARK_FUNCTION;
  #endif
  auto C00 = A[ 0]*B[ 0] + A[ 1]*B[ 6] + A[ 2]*B[12] + A[ 3]*B[18] + A[ 4]*B[24] + A[ 5]*B[30];
  auto C01 = A[ 0]*B[ 1] + A[ 1]*B[ 7] + A[ 2]*B[13] + A[ 3]*B[19] + A[ 4]*B[25] + A[ 5]*B[31];
  auto C02 = A[ 0]*B[ 2] + A[ 1]*B[ 8] + A[ 2]*B[14] + A[ 3]*B[20] + A[ 4]*B[26] + A[ 5]*B[32];
  auto C03 = A[ 0]*B[ 3] + A[ 1]*B[ 9] + A[ 2]*B[15] + A[ 3]*B[21] + A[ 4]*B[27] + A[ 5]*B[33];
  auto C04 = A[ 0]*B[ 4] + A[ 1]*B[10] + A[ 2]*B[16] + A[ 3]*B[22] + A[ 4]*B[28] + A[ 5]*B[34];
  auto C05 = A[ 0]*B[ 5] + A[ 1]*B[11] + A[ 2]*B[17] + A[ 3]*B[23] + A[ 4]*B[29] + A[ 5]*B[35];
  auto C06 = A[ 6]*B[ 0] + A[ 7]*B[ 6] + A[ 8]*B[12] + A[ 9]*B[18] + A[10]*B[24] + A[11]*B[30];
  auto C07 = A[ 6]*B[ 1] + A[ 7]*B[ 7] + A[ 8]*B[13] + A[ 9]*B[19] + A[10]*B[25] + A[11]*B[31];
  auto C08 = A[ 6]*B[ 2] + A[ 7]*B[ 8] + A[ 8]*B[14] + A[ 9]*B[20] + A[10]*B[26] + A[11]*B[32];
  auto C09 = A[ 6]*B[ 3] + A[ 7]*B[ 9] + A[ 8]*B[15] + A[ 9]*B[21] + A[10]*B[27] + A[11]*B[33];
  auto C10 = A[ 6]*B[ 4] + A[ 7]*B[10] + A[ 8]*B[16] + A[ 9]*B[22] + A[10]*B[28] + A[11]*B[34];
  auto C11 = A[ 6]*B[ 5] + A[ 7]*B[11] + A[ 8]*B[17] + A[ 9]*B[23] + A[10]*B[29] + A[11]*B[35];
  auto C12 = A[12]*B[ 0] + A[13]*B[ 6] + A[14]*B[12] + A[15]*B[18] + A[16]*B[24] + A[17]*B[30];
  auto C13 = A[12]*B[ 1] + A[13]*B[ 7] + A[14]*B[13] + A[15]*B[19] + A[16]*B[25] + A[17]*B[31];
  auto C14 = A[12]*B[ 2] + A[13]*B[ 8] + A[14]*B[14] + A[15]*B[20] + A[16]*B[26] + A[17]*B[32];
  auto C15 = A[12]*B[ 3] + A[13]*B[ 9] + A[14]*B[15] + A[15]*B[21] + A[16]*B[27] + A[17]*B[33];
  auto C16 = A[12]*B[ 4] + A[13]*B[10] + A[14]*B[16] + A[15]*B[22] + A[16]*B[28] + A[17]*B[34];
  auto C17 = A[12]*B[ 5] + A[13]*B[11] + A[14]*B[17] + A[15]*B[23] + A[16]*B[29] + A[17]*B[35];
  auto C18 = A[18]*B[ 0] + A[19]*B[ 6] + A[20]*B[12] + A[21]*B[18] + A[22]*B[24] + A[23]*B[30];
  auto C19 = A[18]*B[ 1] + A[19]*B[ 7] + A[20]*B[13] + A[21]*B[19] + A[22]*B[25] + A[23]*B[31];
  auto C20 = A[18]*B[ 2] + A[19]*B[ 8] + A[20]*B[14] + A[21]*B[20] + A[22]*B[26] + A[23]*B[32];
  auto C21 = A[18]*B[ 3] + A[19]*B[ 9] + A[20]*B[15] + A[21]*B[21] + A[22]*B[27] + A[23]*B[33];
  auto C22 = A[18]*B[ 4] + A[19]*B[10] + A[20]*B[16] + A[21]*B[22] + A[22]*B[28] + A[23]*B[34];
  auto C23 = A[18]*B[ 5] + A[19]*B[11] + A[20]*B[17] + A[21]*B[23] + A[22]*B[29] + A[23]*B[35];
  auto C24 = A[24]*B[ 0] + A[25]*B[ 6] + A[26]*B[12] + A[27]*B[18] + A[28]*B[24] + A[29]*B[30];
  auto C25 = A[24]*B[ 1] + A[25]*B[ 7] + A[26]*B[13] + A[27]*B[19] + A[28]*B[25] + A[29]*B[31];
  auto C26 = A[24]*B[ 2] + A[25]*B[ 8] + A[26]*B[14] + A[27]*B[20] + A[28]*B[26] + A[29]*B[32];
  auto C27 = A[24]*B[ 3] + A[25]*B[ 9] + A[26]*B[15] + A[27]*B[21] + A[28]*B[27] + A[29]*B[33];
  auto C28 = A[24]*B[ 4] + A[25]*B[10] + A[26]*B[16] + A[27]*B[22] + A[28]*B[28] + A[29]*B[34];
  auto C29 = A[24]*B[ 5] + A[25]*B[11] + A[26]*B[17] + A[27]*B[23] + A[28]*B[29] + A[29]*B[35];
  auto C30 = A[30]*B[ 0] + A[31]*B[ 6] + A[32]*B[12] + A[33]*B[18] + A[34]*B[24] + A[35]*B[30];
  auto C31 = A[30]*B[ 1] + A[31]*B[ 7] + A[32]*B[13] + A[33]*B[19] + A[34]*B[25] + A[35]*B[31];
  auto C32 = A[30]*B[ 2] + A[31]*B[ 8] + A[32]*B[14] + A[33]*B[20] + A[34]*B[26] + A[35]*B[32];
  auto C33 = A[30]*B[ 3] + A[31]*B[ 9] + A[32]*B[15] + A[33]*B[21] + A[34]*B[27] + A[35]*B[33];
  auto C34 = A[30]*B[ 4] + A[31]*B[10] + A[32]*B[16] + A[33]*B[22] + A[34]*B[28] + A[35]*B[34];
  auto C35 = A[30]*B[ 5] + A[31]*B[11] + A[32]*B[17] + A[33]*B[23] + A[34]*B[29] + A[35]*B[35];
  //trying to force the evaluation all in one place...
  C = MatriplXT66_v2(xt::stack(xt::xtuple(C00, C01, C02, C03, C04, C05, C06, C07, C08, C09,
					  C10, C11, C12, C13, C14, C15, C16, C17, C18, C19,
					  C20, C21, C22, C23, C24, C25, C26, C27, C28, C29,
					  C30, C31, C32, C33, C34, C35),1));
}
