#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"

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

class MatriplXT66_v3
{
  //
public:
  MatriplXT66_v3() { m_ = new xt::xtensor_fixed<float, xt::xshape<NN,36>, xt::layout_type::column_major>(); }
  ~MatriplXT66_v3() {delete m_;}
  //
  auto operator[] (size_t k) { return xt::view(*m_,xt::all(),k); }
  const auto operator[] (size_t k) const { return xt::view(*m_,xt::all(),k); }
  //
  auto operator() (size_t i, size_t j) { return xt::view((*m_),xt::all(),i * 6 + j); }
  const auto operator() (size_t i, size_t j) const { return xt::view((*m_),xt::all(),i * 6 + j); }
  //
  float operator() (size_t i, size_t j, size_t n) const { return (*m_)(n,i * 6 + j); }
  void print(std::ostream& os, size_t n)
  {
    for (size_t i=0; i<6; i++) {
      for (size_t j=0; j<6; j++) {
        os << (*m_)(n,i * 6 + j) << "\t";
      }
      os << std::endl;
    }
  }
  //
 private:
  //
  xt::xtensor_fixed<float, xt::xshape<NN,36>, xt::layout_type::column_major>* m_;
};


void MultiplyXT66(const MatriplXT66& A,
                  const MatriplXT66& B,
                        MatriplXT66& C)
{
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

void MultiplyXT66(const MatriplXT66_v3& A,
                  const MatriplXT66_v3& B,
                        MatriplXT66_v3& C)
{
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

void plainArray_matrix_mult66(const float* Ax, const float* Bx, float* Cx) {
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
}

void plainArray_element_mult66(const float* Ax, const float* Bx, float* Cx) {
  for (size_t x = 0; x < NN; ++x) {
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        for (size_t k = 0; k < 6; ++k) {
          Cx[ x + (i*6 + j)*NN ] += Ax[ x + (i*6 + k)*NN ] * Bx[ x + (k*6 + j)*NN ];
        }
      }
    }
  }
}

void plainArray_el16mx_mult66(const float* Ax, const float* Bx, float* Cx) {
  for (size_t x = 0; x < NN/16; ++x) {
    const size_t Nx = x*16*36;
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
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

void plainArray_el16mx_mult66_v1(const float* Ax, const float* Bx, float* Cx) {
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

void plainArray_xsimd_mult66(const std::vector<float, XSIMD_DEFAULT_ALLOCATOR(float)>* A,
			     const std::vector<float, XSIMD_DEFAULT_ALLOCATOR(float)>* B,
			     std::vector<float, XSIMD_DEFAULT_ALLOCATOR(float)>* C,
			     const size_t vec_size, const size_t inc) {
  //
  using b_type = xsimd::simd_type<float>;
  //
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
  //
}
