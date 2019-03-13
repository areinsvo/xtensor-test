#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

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
    res = xt::pow(arr3, xt::view(arr4, xt::range(0,3)));

    std::cout << res << std::endl;

    arr4.reshape({4, 1});

    std::cout << arr4 << std::endl;

    res = xt::pow(arr3, arr4);

    std::cout << res << std::endl;

    return 0;
}
