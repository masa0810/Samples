#pragma once

MSVC_ALL_WARNING_PUSH

#include <tuple>
#include <type_traits>
#include <vector>

#include <boost/mp11.hpp>

MSVC_WARNING_POP

namespace bmp11 {

enum class Type { Src, Bg, SrcY, BgY, Edge, Flow, BgSub };

template <Type T, typename V, int C>
struct I_ {
  static constexpr auto ImageType = T;
  using ValueType = V;
  static constexpr auto Channel = C;
};

template <typename T, typename... Args>
struct DuplicateCheck_;

template <typename T>
struct DuplicateCheck_<T> {};

template <typename T, typename I, typename... Args>
struct DuplicateCheck_<T, I, Args...>
    : DuplicateCheck_<boost::mp11::mp_push_back<T, std::integral_constant<Type, I::ImageType>>,
                      Args...> {
  static_assert(!boost::mp11::mp_contains<T, std::integral_constant<Type, I::ImageType>>::value);
};

template <std::size_t Idx, typename... Args>
struct ImageCollectionImpl_;

template <std::size_t Idx, typename T, typename V, typename C>
struct ImageCollectionImpl_<Idx, T, V, C> {
  using ImageTypeList = T;    // std::tuple<std::integral_constant<Type, I::ImageType>...>
  using ValueTypeList = V;    // std::tuple<...>
  using ChannelTypeList = C;  // std::tuple<std::integral_constant<int, Channel>...>
  static constexpr auto NumOfImage = Idx;
};

template <std::size_t Idx, typename T, typename V, typename C, typename I, typename... Args>
struct ImageCollectionImpl_<Idx, T, V, C, I, Args...>
    : public ImageCollectionImpl_<
          Idx + 1, boost::mp11::mp_push_back<T, std::integral_constant<Type, I::ImageType>>,
          boost::mp11::mp_push_back<V, typename I::ValueType>,
          boost::mp11::mp_push_back<C, boost::mp11::mp_int<I::Channel>>, Args...> {
  using Next =
      ImageCollectionImpl_<Idx + 1,
                           boost::mp11::mp_push_back<T, std::integral_constant<Type, I::ImageType>>,
                           boost::mp11::mp_push_back<V, typename I::ValueType>,
                           boost::mp11::mp_push_back<C, boost::mp11::mp_int<I::Channel>>, Args...>;
  using ImageTypeList = typename Next::ImageTypeList;
  using ValueTypeList = typename Next::ValueTypeList;
  using ChannelTypeList = typename Next::ChannelTypeList;
  static constexpr auto NumOfImage = Next::NumOfImage;

  using ImageType = std::integral_constant<Type, I::ImageType>;
  //static constexpr auto ImageType = I::ImageType;
  using ValueType = typename I::ValueType;
  static constexpr auto Channel = I::Channel;

 private:
  static constexpr auto Index = Idx;
};

template <typename T, typename L, bool IsFind>
struct FindValueTypeImpl_;

template <typename T, typename L>
struct FindValueTypeImpl_<T, L, true> {
  using ValueType_ = typename L::ValueType;
  static constexpr auto Channel_ = L::Channel;
};

template <typename T, typename L>
struct FindValueTypeImpl_<T, L, false> : FindValueTypeImpl_<T, typename L::Next, std::is_same_v<T, typename L::Next::ImageType>> {
  using Next = FindValueTypeImpl_<T, typename L::Next, std::is_same_v<T, typename L::Next::ImageType>>;
  using ValueType_ = typename Next::ValueType_;
  static constexpr auto Channel_ = Next::Channel_;
};

template <typename T, typename L>
struct FindValueType_ : FindValueTypeImpl_<T, L, std::is_same_v<T, typename L::ImageType>> {
  using Next = FindValueTypeImpl_<T, L, std::is_same_v<T, typename L::ImageType>>;
  using ValueType_ = typename Next::ValueType_;
  static constexpr auto Channel_ = Next::Channel_;
};

template <template <typename...> class Alloc_, typename... Args>
struct ImageCollection_
    : ImageCollectionImpl_<0, std::tuple<>, std::tuple<>, std::tuple<>, Args...>,
      private DuplicateCheck_<std::tuple<>, Args...> {
  using Base = ImageCollectionImpl_<0, std::tuple<>, std::tuple<>, std::tuple<>, Args...>;
  using ImageTypeList = typename Base::ImageTypeList;
  using ValueTypeList = typename Base::ValueTypeList;
  using ChannelTypeList = typename Base::ChannelTypeList;
  static constexpr auto NumOfImage = Base::NumOfImage;

  // vector type
  template <typename T>
  using VecType = std::vector<T, Alloc_<T>>;

  // 型取得
  template <Type T>
  using ValueType_ = typename FindValueType_<std::integral_constant<Type, T>, Base>::ValueType_;
  // チャンネル取得
  template <Type T>
  static constexpr auto Channel_ = FindValueType_<std::integral_constant<Type, T>, Base>::Channel_;
};

}  // namespace bmp11
