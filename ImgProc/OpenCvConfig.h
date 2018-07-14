///////////////////////////////////////////////////
//
// OpenCVの設定とユーティリティ
//
// (c)2015 SECOM.CO.,LTD.
//
///////////////////////////////////////////////////

#pragma once

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <string>
#include <type_traits>
#include <utility>

#include <boost/container_hash/hash.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/xpressive/xpressive.hpp>

#include <opencv2/core.hpp>

// 警告抑制解除
MSVC_WARNING_POP

// インポート・エクスポートマクロ
#ifndef _EXPORT_COMMON_
#if !defined(_MSC_VER) || defined(_LIB)
#define _EXPORT_COMMON_
#else
MSVC_WARNING_DISABLE(251)
#ifdef Common_EXPORTS
#define _EXPORT_COMMON_ __declspec(dllexport)
#else
#define _EXPORT_COMMON_ __declspec(dllimport)
#endif
#endif
#endif

// std
namespace std {

/// <summary>
/// min
/// </summary>
/// <param name="a">[in] 比較対象a</param>
/// <param name="b">[in] 比較対象b</param>
/// <returns>比較対象の内、値の小さい方</returns>
/// <remarks>cv::Point用</remarks>
template <typename T>
cv::Point_<T> min(const cv::Point_<T>& a, const cv::Point_<T>& b) {
  return {min(a.x, b.x), min(a.y, b.y)};
}

/// <summary>
/// max
/// </summary>
/// <param name="a">[in] 比較対象a</param>
/// <param name="b">[in] 比較対象b</param>
/// <returns>比較対象の内、値の大きい方</returns>
/// <remarks>cv::Point用</remarks>
template <typename T>
cv::Point_<T> max(const cv::Point_<T>& a, const cv::Point_<T>& b) {
  return {max(a.x, b.x), max(a.y, b.y)};
}

/// <summary>
/// min
/// </summary>
/// <param name="a">[in] 比較対象a</param>
/// <param name="b">[in] 比較対象b</param>
/// <returns>比較対象の内、値の小さい方</returns>
/// <remarks>cv::Point3 用</remarks>
template <typename T>
cv::Point3_<T> min(const cv::Point3_<T>& a, const cv::Point3_<T>& b) {
  return {min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
}

/// <summary>
/// max
/// </summary>
/// <param name="a">[in] 比較対象a</param>
/// <param name="b">[in] 比較対象b</param>
/// <returns>比較対象の内、値の大きい方</returns>
/// <remarks>cv::Point3 用</remarks>
template <typename T>
cv::Point3_<T> max(const cv::Point3_<T>& a, const cv::Point3_<T>& b) {
  return {max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
}

/// <summary>
/// min
/// </summary>
/// <param name="a">[in] 比較対象a</param>
/// <param name="b">[in] 比較対象b</param>
/// <returns>比較対象の内、値の小さい方</returns>
/// <remarks>cv::Size 用</remarks>
template <typename T>
cv::Size_<T> min(const cv::Size_<T>& a, const cv::Size_<T>& b) {
  return {min(a.width, b.width), min(a.height, b.height)};
}

/// <summary>
/// max
/// </summary>
/// <param name="a">[in] 比較対象a</param>
/// <param name="b">[in] 比較対象b</param>
/// <returns>比較対象の内、値の大きい方</returns>
/// <remarks>cv::Size 用</remarks>
template <typename T>
cv::Size_<T> max(const cv::Size_<T>& a, const cv::Size_<T>& b) {
  return {max(a.width, b.width), max(a.height, b.height)};
}

/// <summary>
/// min
/// </summary>
/// <param name="a">[in] 比較対象a</param>
/// <param name="b">[in] 比較対象b</param>
/// <returns>比較対象の内、値の小さい方</returns>
/// <remarks>cv::Rect 用</remarks>
template <typename T>
cv::Rect_<T> min(const cv::Rect_<T>& a, const cv::Rect_<T>& b) {
  return a & b;
}

/// <summary>
/// max
/// </summary>
/// <param name="a">[in] 比較対象a</param>
/// <param name="b">[in] 比較対象b</param>
/// <returns>比較対象の内、値の大きい方</returns>
/// <remarks>cv::Rect 用</remarks>
template <typename T>
cv::Rect_<T> max(const cv::Rect_<T>& a, const cv::Rect_<T>& b) {
  return a | b;
}

MSVC_WARNING_PUSH

// 装飾された名前の長さが限界警告対策
MSVC_WARNING_DISABLE(503)

/// <summary>
/// 入力ストリーム演算子
/// </summary>
/// <param name="is">[in] 入力ストリーム</param>
/// <param name="point">[out] 比較対象b</param>
/// <returns>ストリーム</returns>
/// <remarks>cv::Point 用</remarks>
template <typename T>
std::istream& operator>>(std::istream& is, cv::Point_<T>& point) {
  namespace bst = boost;
  namespace xp = boost::xpressive;

  std::string tmp;
  std::getline(is, tmp);

  static const xp::sregex r = '[' >> (xp::s1 = +xp::set[xp::range('0', '9') | '-' | '.']) >> ", " >>
                              (xp::s2 = +xp::set[xp::range('0', '9') | '-' | '.']) >> ']';
  xp::smatch what;
  if (xp::regex_match(tmp, what, r)) {
    point.x = bst::lexical_cast<T>(what[1]);
    point.y = bst::lexical_cast<T>(what[2]);
  }

  return is;
}

/// <summary>
/// 入力ストリーム演算子
/// </summary>
/// <param name="is">[in] 入力ストリーム</param>
/// <param name="point">[out] 比較対象b</param>
/// <returns>ストリーム</returns>
/// <remarks>cv::Point3 用</remarks>
template <typename T>
std::istream& operator>>(std::istream& is, cv::Point3_<T>& point) {
  namespace bst = boost;
  namespace xp = boost::xpressive;

  std::string tmp;
  std::getline(is, tmp);

  static const xp::sregex r = '[' >> (xp::s1 = +xp::set[xp::range('0', '9') | '-' | '.']) >> ", " >>
                              (xp::s2 = +xp::set[xp::range('0', '9') | '-' | '.']) >> ", " >>
                              (xp::s3 = +xp::set[xp::range('0', '9') | '-' | '.']) >> ']';
  xp::smatch what;
  if (xp::regex_match(tmp, what, r)) {
    point.x = bst::lexical_cast<T>(what[1]);
    point.y = bst::lexical_cast<T>(what[2]);
    point.z = bst::lexical_cast<T>(what[3]);
  }

  return is;
}

/// <summary>
/// 入力ストリーム演算子
/// </summary>
/// <param name="is">[in] 入力ストリーム</param>
/// <param name="size">[out] 比較対象b</param>
/// <returns>ストリーム</returns>
/// <remarks>cv::Size 用</remarks>
template <typename T>
std::istream& operator>>(std::istream& is, cv::Size_<T>& size) {
  namespace bst = boost;
  namespace xp = boost::xpressive;

  std::string tmp;
  std::getline(is, tmp);

  static const xp::sregex r = '[' >> (xp::s1 = +xp::set[xp::range('0', '9') | '-' | '.']) >>
                              " x " >> (xp::s2 = +xp::set[xp::range('0', '9') | '-' | '.']) >> ']';
  xp::smatch what;
  if (xp::regex_match(tmp, what, r)) {
    size.width = bst::lexical_cast<T>(what[1]);
    size.height = bst::lexical_cast<T>(what[2]);
  }

  return is;
}

/// <summary>
/// 入力ストリーム演算子
/// </summary>
/// <param name="is">[in] 入力ストリーム</param>
/// <param name="r">[out] 比較対象b</param>
/// <returns>ストリーム</returns>
/// <remarks>cv::Rect 用</remarks>
template <typename T>
std::istream& operator>>(std::istream& is, cv::Rect_<T>& rect) {
  namespace bst = boost;
  namespace xp = boost::xpressive;

  std::string tmp;
  std::getline(is, tmp);

  static const xp::sregex r = '[' >> (xp::s1 = +xp::set[xp::range('0', '9') | '-' | '.']) >>
                              " x " >> (xp::s2 = +xp::set[xp::range('0', '9') | '-' | '.']) >>
                              " from (" >>
                              (xp::s3 = ! - +xp::set[xp::range('0', '9') | '-' | '.']) >> ", " >>
                              (xp::s4 = ! - +xp::set[xp::range('0', '9') | '-' | '.']) >> ")]";
  xp::smatch what;
  if (xp::regex_match(tmp, what, r)) {
    rect.width = bst::lexical_cast<T>(what[1]);
    rect.height = bst::lexical_cast<T>(what[2]);
    rect.x = bst::lexical_cast<T>(what[3]);
    rect.y = bst::lexical_cast<T>(what[4]);
  }

  return is;
}

MSVC_WARNING_POP

}  // namespace std

// シリアライズ設定
namespace boost { namespace serialization {

/// <summary>
/// シリアライズ関数
/// </summary>
/// <param name="ar">[in] アーカイバ</param>
/// <param name="p">[out] 対象</param>
/// <remarks>cv::Point 用</remarks>
template <class Archive, typename T>
void serialize(Archive& ar, cv::Point_<T>& p, const unsigned) {
  ar& make_nvp("X", p.x);
  ar& make_nvp("Y", p.y);
}

/// <summary>
/// シリアライズ関数
/// </summary>
/// <param name="ar">[in] アーカイバ</param>
/// <param name="p">[out] 対象</param>
/// <remarks>cv::Point3 用</remarks>
template <class Archive, typename T>
void serialize(Archive& ar, cv::Point3_<T>& p, const unsigned) {
  ar& make_nvp("X", p.x);
  ar& make_nvp("Y", p.y);
  ar& make_nvp("Z", p.z);
}

/// <summary>
/// シリアライズ関数
/// </summary>
/// <param name="ar">[in] アーカイバ</param>
/// <param name="s">[out] 対象</param>
/// <remarks>cv::Size 用</remarks>
template <class Archive, typename T>
void serialize(Archive& ar, cv::Size_<T>& s, const unsigned) {
  ar& make_nvp("Height", s.height);
  ar& make_nvp("Width", s.width);
}

/// <summary>
/// シリアライズ関数
/// </summary>
/// <param name="ar">[in] アーカイバ</param>
/// <param name="r">[out] 対象</param>
/// <remarks>cv::Rect 用</remarks>
template <class Archive, typename T>
void serialize(Archive& ar, cv::Rect_<T>& r, const unsigned) {
  ar& make_nvp("X", r.x);
  ar& make_nvp("Y", r.y);
  ar& make_nvp("Width", r.width);
  ar& make_nvp("Height", r.height);
}

}}  // namespace boost::serialization

// OpenCVユーティリティ
namespace cv {

// タイプを取得
template <typename T>
constexpr int ValueType() noexcept {
  return CV_USRTYPE1;
}
template <>
constexpr int ValueType<std::uint8_t>() noexcept {
  return CV_8U;
}
template <>
constexpr int ValueType<std::int8_t>() noexcept {
  return CV_8S;
}
template <>
constexpr int ValueType<std::uint16_t>() noexcept {
  return CV_16U;
}
template <>
constexpr int ValueType<std::int16_t>() noexcept {
  return CV_16S;
}
template <>
constexpr int ValueType<std::int32_t>() noexcept {
  return CV_32S;
}
template <>
constexpr int ValueType<float>() noexcept {
  return CV_32F;
}
template <>
constexpr int ValueType<double>() noexcept {
  return CV_64F;
}
template <typename T>
constexpr int GetType(const int ch = 1) noexcept {
  return CV_MAKETYPE(ValueType<T>(), ch);
}
template <typename T, int Ch = 1>
constexpr int Type() noexcept {
  return GetType<T>(Ch);
}

// 任意の型のポインタにバイト単位の加算
template <typename T>
const T* PtrByteInc(const T* ptr, const int n) {
  return reinterpret_cast<const T*>(reinterpret_cast<const std::uint8_t*>(ptr) + n);
}

// 任意の型のポインタにバイト単位の加算
template <typename T>
T* PtrByteInc(T* ptr, const int n) {
  return reinterpret_cast<T*>(reinterpret_cast<std::uint8_t*>(ptr) + n);
}

// ブレンド関数
template <typename Td, typename Ts, typename Tb, typename Tf, int N,
          std::enable_if_t<std::is_floating_point<Tf>::value, std::nullptr_t> = nullptr>
cv::Vec<Td, N> Blend(const cv::Vec<Ts, N>& src, const cv::Vec<Tb, N>& blend, const Tf blendRatio) {
  const auto ratioInv = Tf(1) - blendRatio;
  cv::Vec<Td, N> ret;
  for (auto i = 0; i < N; ++i)
    ret[i] = saturate_cast<Td>(static_cast<Tf>(src[i]) * ratioInv +
                               static_cast<Tf>(blend[i]) * blendRatio);
  return ret;
}

// ブレンド関数
template <typename T1, typename T2, int N,
          std::enable_if_t<std::is_floating_point<T2>::value, std::nullptr_t> = nullptr>
cv::Vec<T1, N> Blend(const cv::Vec<T1, N>& src, const cv::Vec<T1, N>& blend, const T2 blendRatio) {
  return Blend<T1, T1, T1, T2, N>(src, blend, blendRatio);
}

// コピー関数
template <typename T1, typename T2,
          std::enable_if_t<std::is_same<T1, T2>::value, std::nullptr_t> = nullptr>
void Copy(const cv::Mat& src, cv::Mat& dst) {
  src.copyTo(dst);
}

// コピー関数
template <typename T1, typename T2,
          std::enable_if_t<!std::is_same<T1, T2>::value, std::nullptr_t> = nullptr>
void Copy(const cv::Mat& src, cv::Mat& dst) {
  src.convertTo(dst, Type<T2>());
}

// コピー関数
template <typename T>
void Copy(const cv::Mat& src, cv::Mat& dst) {
  switch (CV_MAT_DEPTH(dst.type())) {
    case CV_8U:
      Copy<T, std::uint8_t>(src, dst);
      break;
    case CV_8S:
      Copy<T, std::int8_t>(src, dst);
      break;
    case CV_16U:
      Copy<T, std::uint16_t>(src, dst);
      break;
    case CV_16S:
      Copy<T, std::int16_t>(src, dst);
      break;
    case CV_32S:
      Copy<T, std::int32_t>(src, dst);
      break;
    case CV_32F:
      Copy<T, float>(src, dst);
      break;
    case CV_64F:
      Copy<T, double>(src, dst);
      break;
    // 例外処理
    default:
      break;
  }
}

// コピー関数
_EXPORT_COMMON_ void Copy(const cv::Mat& src, cv::Mat& dst);

// コピー関数(Gray->RGB)
template <typename T1, typename T2>
void CopyGrayToBgr(const cv::Mat& src, cv::Mat& dst) {
  for (auto y = 0; y < src.rows; ++y) {
    auto x = 0;
    const auto* ptrSrc = src.ptr<T1>(y, x);
    auto* ptrDst = dst.ptr<Vec<T2, 3>>(y, x);
    for (; x < src.cols; ++x, ++ptrSrc, ++ptrDst) {
      auto& pix = *ptrDst;
      pix[0] = pix[1] = pix[2] = saturate_cast<T2>(*ptrSrc);
    }
  }
}

}  // namespace cv

namespace boost {

/// <summary>
/// ハッシュ関数
/// </summary>
/// <param name="p">[in] ハッシュ計算対象</param>
/// <returns>ハッシュ値</returns>
template <typename T>
std::size_t hash_value(const cv::Point_<T>& p) {
  auto hash = boost::hash<T>()(p.x);
  hash_combine(hash, p.y);
  return hash;
}

/// <summary>
/// ハッシュ関数
/// </summary>
/// <param name="p">[in] ハッシュ計算対象</param>
/// <returns>ハッシュ値</returns>
template <typename T>
std::size_t hash_value(const cv::Point3_<T>& p) {
  auto hash = boost::hash<T>()(p.x);
  hash_combine(hash, p.y);
  hash_combine(hash, p.z);
  return hash;
}

/// <summary>
/// ハッシュ関数
/// </summary>
/// <param name="s">[in] ハッシュ計算対象</param>
/// <returns>ハッシュ値</returns>
template <typename T>
std::size_t hash_value(const cv::Size_<T>& s) {
  auto hash = boost::hash<T>()(s.height);
  hash_combine(hash, s.width);
  return hash;
}

/// <summary>
/// ハッシュ関数
/// </summary>
/// <param name="r">[in] ハッシュ計算対象</param>
/// <returns>ハッシュ値</returns>
template <typename T>
std::size_t hash_value(const cv::Rect_<T>& r) {
  auto hash = boost::hash<T>()(r.x);
  hash_combine(hash, r.y);
  hash_combine(hash, r.width);
  hash_combine(hash, r.height);
  return hash;
}

}  // namespace boost

// 全組み合わせループ
#define CV_MAT_TYPE_LOOP(name, IMPL)                                                         \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                                             \
      IMPL, ((name))((std::uint8_t)(std::int8_t)(std::uint16_t)(std::int16_t)(std::int32_t)( \
                float)(double))((1)(2)(3)(4)))

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, std::nullptr_t> = nullptr>
cv::Point cvFloor(const cv::Point_<T>& p) {
  return {cvFloor(p.x), cvFloor(p.y)};
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, std::nullptr_t> = nullptr>
const cv::Point_<T>& cvFloor(const cv::Point_<T>& p) {
  return p;
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, std::nullptr_t> = nullptr>
cv::Point3i cvFloor(const cv::Point3_<T>& p) {
  return {cvFloor(p.x), cvFloor(p.y), cvFloor(p.z)};
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, std::nullptr_t> = nullptr>
const cv::Point3_<T>& cvFloor(const cv::Point3_<T>& p) {
  return p;
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, std::nullptr_t> = nullptr>
cv::Point cvCeil(const cv::Point_<T>& p) {
  return {cvCeil(p.x), cvCeil(p.y)};
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, std::nullptr_t> = nullptr>
const cv::Point_<T>& cvCeil(const cv::Point_<T>& p) {
  return p;
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, std::nullptr_t> = nullptr>
cv::Point3i cvCeil(const cv::Point3_<T>& p) {
  return {cvCeil(p.x), cvCeil(p.y), cvCeil(p.z)};
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, std::nullptr_t> = nullptr>
const cv::Point3_<T>& cvCeil(const cv::Point3_<T>& p) {
  return p;
}
