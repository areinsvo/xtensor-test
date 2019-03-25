#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"

#include "xtensor/xvectorize.hpp"
#include "xtensor/xfixed.hpp"

int f(int a, int b)
{
  return a + 2 * b;
}

int main(int argc, char* argv[])
{
    xt::xarray<float> arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};

    xt::xarray<float> arr2
      {5.0, 6.0, 7.0};

    xt::xarray<float> res = xt::view(arr1, 1) + arr2;

    std::cout << res << std::endl;

    std::cout << arr1(1,2) << std::endl;

    arr1.reshape({1,9});

    std::cout << arr1 << std::endl;

    std::cout << arr1(5) << std::endl;

    xt::xarray<float> arr3
      {1.0, 2.0, 3.0};

    xt::xarray<unsigned int> arr4
      {4, 5, 6, 7};

    std::cout << xt::view(arr4, xt::range(0,3)) << std::endl;

    // warning: pow seem not to work with xsimd
    // res = xt::pow(arr3, xt::view(arr4, xt::range(0,3)));

    std::cout << res << std::endl;

    arr4.reshape({4, 1});

    std::cout << arr4 << std::endl;

    // res = xt::pow(arr3, arr4);

    std::cout << res << std::endl;

    xt::xtensor<float, 2> tsr1
      {{1.0, 2.0, 3.0},
       {4.0, 5.0, 6.0}};

    for (auto& el : tsr1.shape()) {std::cout << el << ", "; }
    std::cout << std::endl;
    std::cout << tsr1 << std::endl;
    std::cout << tsr1(0,1) << std::endl;
    std::cout << tsr1(1,2) << std::endl;
    std::cout << xt::view(tsr1,xt::all(),1) << std::endl;

    xt::xarray<float> tsr2
      {{1.0, 2.0, 3.0},
       {4.0, 5.0, 6.0}};

    for (auto& el : tsr2.shape()) {std::cout << el << ", "; }
    std::cout << std::endl;
    std::cout << tsr2 << std::endl;
    std::cout << tsr2(0,1) << std::endl;
    std::cout << tsr2(1,2) << std::endl;
    std::cout << xt::view(tsr2,xt::all(),1) << std::endl;

    std::vector<size_t> shape = {10, 3};
    xt::xarray<float> tsr3(shape);
    // xt::view(tsr3,0,xt::all()) = xt::xarray<float>({1.0, 2.0, 3.0});
    // xt::view(tsr3,1,xt::all()) = xt::xarray<float>({4.0, 5.0, 6.0});
    float count = 0;
    for (size_t i=0;i<3;++i) xt::view(tsr3,xt::all(),i) = xt::random::randn<float>({10});

    for (auto& el : tsr3.shape()) {std::cout << el << ", "; }
    std::cout << std::endl;
    std::cout << tsr3 << std::endl;
    std::cout << tsr3(0,1) << std::endl;
    std::cout << tsr3(1,2) << std::endl;
    std::cout << xt::view(tsr3,xt::all(),1) << std::endl;

    auto vecf = xt::vectorize(f);
    xt::xarray<int> a = { 11, 12, 13 };
    xt::xarray<int> b = {  1,  2,  3 };
    xt::xarray<int> res_f = vecf(a, b);
    std::cout << "res_f\n" << res_f << std::endl;

    xt::xarray<int> arra{{1, 2, 3},{4, 5, 6},{7, 8, 9}};
    xt::xarray<int> arrb{{1, 2, 3},{4, 5, 6},{7, 8, 9}};
    xt::xarray<int> res_af = vecf(xt::view(arra,0,xt::all()), xt::view(arrb,0,xt::all()));
    std::cout << res_af << std::endl;

    return 0;
}
