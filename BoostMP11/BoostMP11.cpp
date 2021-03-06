#include "BoostMP11.h"

MSVC_ALL_WARNING_PUSH

#include <cstdint>

#include <iostream>
#include <memory>

#include <boost/type_index.hpp>

MSVC_WARNING_POP

using Images = bmp11::ImageCollection_<
    // アロケーター
    std::allocator,
    // Src
    bmp11::I_<bmp11::Type::Src, std::uint8_t, 3>,
    // Bg
    bmp11::I_<bmp11::Type::Bg, std::int8_t, 3>,
    // SrcY
    bmp11::I_<bmp11::Type::SrcY, std::uint16_t, 1>,
    // BgY
    bmp11::I_<bmp11::Type::BgY, std::int16_t, 1>,
    // Edge
    bmp11::I_<bmp11::Type::Edge, std::int32_t, 2>,
    // Flow
    bmp11::I_<bmp11::Type::Flow, float, 2>,
    // BgSub
    bmp11::I_<bmp11::Type::BgSub, double, 4>>;

template <typename T>
void f() {
  std::cout << boost::typeindex::type_id_with_cvr<T>().pretty_name() << std::endl;
}

int main() {
  f<Images>();
  f<Images::ValueTypeList>();
  f<Images::ChannelTypeList>();
  f<Images::ImageType>();
  std::cout << Images::NumOfImage << std::endl;

  f<Images::ValueType_<bmp11::Type::Edge>>();
  std::cout << Images::Channel_<bmp11::Type::BgSub> << std::endl;

  return 0;
}
