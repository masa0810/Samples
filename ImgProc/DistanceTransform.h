#pragma once

#include "ImageCollection.h"
#include "NonCopyMembers.h"
#include "OpenCvConfig.h"
#include "TbbConfig.h"

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <cmath>
#include <cstddef>

#include <algorithm>
#include <functional>
#include <limits>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <vector>

#include <boost/container_hash/hash.hpp>
#include <boost/operators.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>

#include <opencv2/core.hpp>

#include <tbb/tbb.h>

// 警告抑制解除
MSVC_WARNING_POP

/// <summary>
/// 画像処理
/// </summary>
namespace imgproc {

/// <summary>
/// 詳細
/// </summary>
namespace detail {

/// <summary>
/// 計算テーブル
/// </summary>
/// <remarks>インデックスの二乗などの事前計算可能な計算テーブル</remarks>
/// @tparam T 計算テーブルの型
/// @tparam Alloc_ テーブルのアロケータ
template <typename T, template <typename...> class Alloc_>
struct Table_ {
  //! ベクトルタイプ
  //!@ U コンテナの中身の型
  template <typename U>
  using VecType = std::vector<U, Alloc_<U>>;
  //! テーブルサイズ
  int Size = 0;
  //! 二乗テーブル
  VecType<T> Sqr = {};
  //! 飽和分岐テーブル
  VecType<int> Sat = {};
  //! 逆数テーブル
  VecType<T> Inv = {};
  //! 飽和分岐テーブルのスタートインデックス
  int SatStart = 0;

#pragma region デフォルトメソッド定義

  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  Table_() = default;

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  Table_(const Table_&) = default;

  /// <summary>
  /// 代入演算子
  /// </summary>
  Table_& operator=(const Table_&) = default;

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  Table_(Table_&&) = default;

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  Table_& operator=(Table_&&) = default;

  /// <summary>
  /// デストラクタ
  /// </summary>
  ~Table_() = default;

#pragma endregion

  /// <summary>
  /// 初期化
  /// </summary>
  /// <param name="size">テーブルサイズ</param>
  void Init(const cv::Size& size) {
    this->Size = std::max(size.width, size.height);

    this->Sqr.resize(this->Size << 1);
    this->Sat.resize(this->Size * 3 + 1);
    this->Inv.resize(this->Size);
    SatStart = (this->Size << 1) + 1;

    auto* ptrSqr0 = &this->Sqr[0];
    auto* ptrSqr1 = ptrSqr0 + this->Size;
    auto* ptrSat0 = &this->Sat[0];
    auto* ptrSat1 = ptrSat0 + this->Size;
    auto* ptrSat2 = ptrSat1 + this->Size;
    auto* ptrInv = &this->Inv[0];
    for (auto i = 0; i < this->Size;
         ++i, ++ptrSqr0, ++ptrSqr1, ++ptrSat0, ++ptrSat1, ++ptrSat2, ++ptrInv) {
      *ptrSqr0 = i * i;
      *ptrSqr1 = std::numeric_limits<T>::max();
      *ptrSat0 = *ptrSat1 = 0;
      *ptrSat2 = i;
      *ptrInv = T(0.5) / i;
    }
    *ptrSat2 = this->Size;
  }

 private:
#pragma region friend宣言

  /// <summary>
  /// ハッシュ関数のフレンド宣言
  /// </summary>
  /// <param name="rhs">[in] ハッシュを計算する対象</param>
  /// <returns>ハッシュ値</returns>
  /// <remarks>boost::hash でハッシュ値を取得出来るようにする。</remarks>
  friend std::size_t hash_value(const Table_& rhs) {
    auto hash = boost::hash_value(rhs.Size);
    boost::hash_combine(hash, rhs.Sqr);
    boost::hash_combine(hash, rhs.Sat);
    boost::hash_combine(hash, rhs.Inv);
    boost::hash_combine(hash, rhs.SatStart);
    return hash;
  }

  /// <summary>
  /// 等値比較演算子のフレンド宣言
  /// </summary>
  /// <param name="lhs">[in] 比較対象</param>
  /// <param name="rhs">[in] 比較対象</param>
  /// <returns>比較結果</returns>
  friend bool operator==(const Table_& lhs, const Table_& rhs) {
    return hash_value(lhs) == hash_value(rhs);
  }

  /// <summary>
  /// 等値比較演算子のフレンド宣言
  /// </summary>
  /// <param name="lhs">[in] 比較対象</param>
  /// <param name="rhs">[in] 比較対象</param>
  /// <returns>比較結果</returns>
  friend bool operator!=(const Table_& lhs, const Table_& rhs) { return !(lhs == rhs); }

  /// <summary>
  /// ストリーム出力関数のフレンド宣言
  /// </summary>
  /// <param name="os">[in] 出力ストリーム</param>
  /// <param name="rhs">[in] 出力オブジェクト</param>
  /// <returns>ストリーム</returns>
  template <typename charT, typename traits>
  friend std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os,
                                                       const Table_& rhs) {
    os << hash_value(rhs);
    return os;
  }

  //! @name シリアライズ用設定
  //@{
  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned) {
    ar& BOOST_SERIALIZATION_NVP(Size);
    ar& BOOST_SERIALIZATION_NVP(Sqr);
    ar& BOOST_SERIALIZATION_NVP(Sat);
    ar& BOOST_SERIALIZATION_NVP(Inv);
    ar& BOOST_SERIALIZATION_NVP(SatStart);
  }
  //@}

#pragma endregion
};

/// <summary>
/// スレッド・ローカル・ストレージのバッファ取得
/// </summary>
/// <returns>0:インデックスバッファ、1:一時バッファ</returns>
/// <remarks>TLSの利用で並列化時のバッファの効率化に期待</remarks>
/// @tparam T バッファの型
/// @tparam Alloc_ バッファのアロケータ
template <typename T, template <typename...> class Alloc_>
std::tuple<std::vector<int, Alloc_<int>>&, std::vector<T, Alloc_<T>>&> GetTlsBuffer(
    const int bufSize) {
  using IdxBufType = std::vector<int, Alloc_<int>>;
  using TmpBufType = std::vector<T, Alloc_<T>>;
  thread_local auto _bufSize = bufSize;
  thread_local IdxBufType bufIdx(bufSize);
  thread_local TmpBufType bufTmp(bufSize << 1);
  // バッファサイズが小さい場合はリサイズ
  if (_bufSize < bufSize) {
    _bufSize = bufSize;
    bufIdx.resize(_bufSize);
    bufTmp.resize(_bufSize << 1);
  }
  return std::make_tuple(std::ref(bufIdx), std::ref(bufTmp));
}

/// <summary>
/// 単純な1次元の距離変換
/// </summary>
/// <param name="sqrTbl">[in] 二乗テーブル</param>
/// <param name="satTbl">[in] 飽和判定テーブル</param>
/// <param name="size">[in] サイズ</param>
/// <param name="byteStep">[in] ステップ(byte)</param>
/// <param name="s">[in] 入力データのポインタ</param>
/// <param name="ptrD0">[out] 出力データのポインタ</param>
/// <param name="ptrI0">[out] 最短インデックスのポインタ</param>
/// <param name="v">[out] インデックスバッファのポインタ</param>
/// <param name="binarization">[in] 二値化関数</param>
/// <remarks>
/// <para>初回のシンプルな距離計算</para>
/// <para>効率化のために、入力画像の二値化処理も同時に行う。</para>
/// </remarks>
/// @tparam T 出力・一時バッファの型
/// @tparam Ts 入力の型
/// @tparam F 入力を二値化する関数
template <typename T, typename Ts, typename F>
void TransformSimple1D(const T* const sqrTbl, const int* const satTbl, const int size,
                       const int byteStep, const Ts* const s, T* ptrD0, int* ptrV0,
                       const F& binarization) {
  const auto sizeM1 = size - 1;
  const auto* ptrS = s + sizeM1;
  auto* ptrV = ptrV0 + sizeM1;
  auto dist = sizeM1;
  for (auto i = sizeM1; i >= 0; --i, --ptrS, --ptrV) {
    *ptrV = dist = binarization(*ptrS) * (dist + 1);
  }
  dist = sizeM1;
  for (auto i = 0; i < size; ++i, ++ptrV0, ptrD0 = cv::PtrByteInc(ptrD0, byteStep)) {
    *ptrV0 = dist = dist + 1 - satTbl[dist - *ptrV0];
    *ptrD0 = sqrTbl[dist];
  }
}

/// <summary>
/// 単純な1次元の距離変換(並列実行対応)
/// </summary>
/// <param name="tbl">[in] 計算テーブル</param>
/// <param name="size">[in] サイズ</param>
/// <param name="byteStep">[in] ステップ(byte)</param>
/// <param name="s">[in] 入力データのポインタ</param>
/// <param name="ptrD0">[out] 出力データのポインタ</param>
/// <param name="ptrI0">[out] 最短インデックスのポインタ</param>
/// <param name="binarization">[in] 二値化関数</param>
/// <remarks>
/// <para>距離変換処理を並列実行するために、TLSバッファを使用する。</para>
/// <para>そのため、この関数は並列スレッドの中で呼び出さなければならない。</para>
/// </remarks>
/// @tparam T 出力・一時バッファの型
/// @tparam Alloc_ バッファのアロケータ
/// @tparam Ts 入力の型
/// @tparam F 入力を二値化する関数
template <typename T, template <typename...> class Alloc_, typename Ts, typename F>
void TransformSimple1D(const Table_<T, Alloc_>& tbl, const int size, const int byteStep,
                       const Ts* const s, T* ptrD0, const F& binarization) {
  auto buf = GetTlsBuffer<T, Alloc_>(tbl.Size);
  auto& bufIdx = std::get<0>(buf);
  TransformSimple1D<T, Ts, F>(&tbl.Sqr[0], &tbl.Sat[tbl.SatStart], size, byteStep, s, ptrD0,
                              &bufIdx[0], binarization);
}

/// <summary>
/// 1次元の距離変換
/// </summary>
/// <param name="sqrTbl">[in] 二乗テーブル</param>
/// <param name="invTbl">[in] 逆数テーブル</param>
/// <param name="size">[in] サイズ</param>
/// <param name="byteStep">[in] ステップ(byte)</param>
/// <param name="f">[in] 入力データのポインタ</param>
/// <param name="ptrD0">[out] 出力データのポインタ</param>
/// <param name="ptrV0">[out] インデックスバッファのポインタ</param>
/// <param name="ptrZ0">[out] 一時バッファのポインタ</param>
/// <param name="conv">[in] 出力変換関数</param>
/// <remarks>
/// <para>2回目の距離計算</para>
/// <para>効率化のために、L2距離から出力値への変換処理も同時に行う。</para>
/// </remarks>
/// @tparam T 入力・一時バッファの型
/// @tparam Td 出力の型
/// @tparam F L2距離を出力値に変換する関数
template <typename T, typename Td, typename F>
void Transform1D(const T* const sqrTbl, const T* const invTbl, const int size, const int byteStep,
                 const T* const f, Td* ptrD0, int* ptrV0, T* ptrZ0, const F& conv) {
  const auto* ptrF1 = f + 1;
  auto* ptrV = ptrV0;
  auto* ptrZ1 = ptrZ0 + 1;

  *ptrV = 0;
  *ptrZ0 = std::numeric_limits<T>::lowest();
  *ptrZ1 = std::numeric_limits<T>::max();
  for (auto q = 1; q < size; ++q, ++ptrF1) {
    const auto A = (*ptrF1) + (q * q);
    const auto v1 = *ptrV;
    auto s = (A - (f[v1] + sqrTbl[v1])) * invTbl[q - v1];
    while (s <= *ptrZ0) {
      --ptrV;
      --ptrZ0;

      const auto v2 = *ptrV;
      s = (A - (f[v2] + sqrTbl[v2])) * invTbl[q - v2];
    }
    ++ptrV;
    ++ptrZ0;

    *ptrV = q;
    *ptrZ0 = s;
    *(ptrZ0 + 1) = std::numeric_limits<T>::max();
  }

  for (auto q = 0; q < size; ++q, ptrD0 = cv::PtrByteInc(ptrD0, byteStep)) {
    while (*ptrZ1 < q) {
      ++ptrV0;
      ++ptrZ1;
    }
    const auto v0 = *ptrV0;
    *ptrD0 = cv::saturate_cast<Td>(conv(sqrTbl[std::abs(q - v0)] + f[v0]));
  }
}

/// <summary>
/// 1次元の距離変換(並列実行対応)
/// </summary>
/// <param name="tbl">[in] 計算テーブル</param>
/// <param name="size">[in] サイズ</param>
/// <param name="byteStep">[in] ステップ(byte)</param>
/// <param name="f">[in] 入力データのポインタ</param>
/// <param name="ptrD0">[out] 出力データのポインタ</param>
/// <param name="conv">[in] 出力変換関数</param>
/// <remarks>距離変換処理を並列実行するために、スレッド毎のバッファを用意する。</remarks>
/// @tparam T 入力・一時バッファの型
/// @tparam Alloc_ バッファのアロケータ
/// @tparam Td 出力の型
/// @tparam F L2距離を出力値に変換する関数
template <typename T, template <typename...> class Alloc_, typename Td, typename F>
void Transform1D(const Table_<T, Alloc_>& tbl, const int size, const int byteStep, const T* const f,
                 T* ptrD0, const F& conv) {
  auto buf = GetTlsBuffer<T, Alloc_>(tbl.Size);
  auto& bufIdx = std::get<0>(buf);
  auto& bufTmp = std::get<1>(buf);
  Transform1D<T>(&tbl.Sqr[0], &tbl.Inv[0], size, byteStep, f, ptrD0, &bufIdx[0], &bufTmp[size],
                 conv);
}

/// <summary>
/// 2次元の距離変換
/// </summary>
/// <param name="tbl">[in] 計算テーブル</param>
/// <param name="src">[in] 入力</param>
/// <param name="dst">[out] 出力</param>
/// <param name="bufSqDist">[out] 二乗距離計算用バッファ</param>
/// <param name="paX">[in] パーティショナ(x方向計算用)</param>
/// <param name="paY">[in] パーティショナ(y方向計算用)</param>
/// <param name="binarization">[in] 二値化関数</param>
/// <param name="conv">[in] 出力変換関数</param>
/// <remarks>
/// <para>距離変換アルゴリズムの特性上、1行または1列毎に距離計算を行う。</para>
/// <para>この実装では、先に列方向(x方向)で距離計算を行ってから行方向(y方向)の距離計算を行う。</para>
/// <para>高速化のために方向毎に処理を並列化するが、最初の並列化(x方向)の際には</para>
/// <para>入力画像の2値化処理、二回目の並列化(y方向)の際には、出力値変換処理を同時に行う。</para>
/// <para>また、距離計算の際に算出される最短インデックスを保持しておくことにより、</para>
/// <para>後から最短座標を計算できるようになっている。</para>
/// </remarks>
/// @tparam T 一時バッファの型
/// @tparam Alloc_ バッファのアロケータ
/// @tparam P パーティショナの型
/// @tparam Ts 入力画像の型
/// @tparam Td 出力画像の型
/// @tparam Fb 2値化関数オブジェクトの型
/// @tparam Fc L2距離を出力値に変換する関数
template <typename T, template <typename...> class Alloc_, typename P, typename Ts, typename Td,
          typename Fb, typename Fc>
void Transform2D(const Table_<T, Alloc_>& tbl, const cv::Mat& src, cv::Mat& dst, cv::Mat bufSqDist,
                 P& paX, P& paY, const Fb& binarization, const Fc& conv) {
  // 行単位で並列化
  tbb::tbb_for(0, src.rows,
               [tbl, size = src.cols, byteStep = bufSqDist.step[0], &src, &bufSqDist,
                &binarization](const auto y) {
                 const auto* const ptrSrc = src.ptr<Ts>(y);
                 auto* ptrBufSqDist = bufSqDist.ptr<T>(0, y);
                 TransformSimple1D<T, Alloc_, Ts, Fb>(tbl, size, byteStep, ptrSrc, ptrBufSqDist,
                                                      binarization);
               },
               paX);

  // 列単位で並列化
  tbb::tbb_for(
      0, dst.cols,
      [tbl, size = dst.rows, byteStep = dst.step[0], &dst, &bufSqDist, &conv](const auto x) {
        const auto* const ptrBufSqDist = bufSqDist.ptr<T>(x);
        auto* ptrDst = dst.ptr<T>(0, x);
        Transform1D<T, Alloc_, Td, Fc>(tbl, size, byteStep, ptrBufSqDist, ptrDst, conv);
      },
      paY);
}

}  // namespace detail

/// <summary>
/// 距離変換画像計算クラス
/// </summary>
/// <remarks>
/// <para>アルゴリズム詳細は以下参照</para>
/// <para>Pedro Felzenszwalb and Daniel Huttenlocher. Distance transforms of sampled functions.
/// Technical report, Cornell University, 2004.</para>
/// </remarks>
/// @tparam Tf 内部計算の型
/// @tparam P パーティショナの型
/// @tparam Alloc_ バッファのアロケータ
template <typename Tf = float, typename P = tbb::affinity_partitioner,
          template <typename...> class Alloc_ = tbb::cache_aligned_allocator>
class DistanceTransform_ : public commonutility::NonCopyMembers_<P, P>,
                           boost::equality_comparable<DistanceTransform_<Tf, P, Alloc_>,
                                                      DistanceTransform_<Tf, P, Alloc_>> {
  static_assert(std::is_floating_point<Tf>{}, "template parameter Tf must be floatng point type");

 public:
  //! 浮動小数点タイプ
  using FloatType = Tf;
  //! パーティショナタイプ
  using PartitionerType = P;

  //! バッファタイプ
  using BufType = commonutility::ImageBufferImpl_<Alloc_, FloatType>;

 private:
  //! テーブルタイプ
  using TableType = detail::Table_<FloatType, Alloc_>;

  //! 計算テーブル
  TableType m_table = {};
  //! 計算バッファ
  BufType m_imgBuf = {};

#pragma region friend宣言

  /// <summary>
  /// ハッシュ関数のフレンド宣言
  /// </summary>
  /// <param name="rhs">[in] ハッシュを計算する対象</param>
  /// <returns>ハッシュ値</returns>
  /// <remarks>boost::hash でハッシュ値を取得出来るようにする。</remarks>
  friend std::size_t hash_value(const DistanceTransform_& rhs) {
    auto hash = hash_value(rhs.m_table);
    boost::hash_combine(hash, rhs.m_imgBuf);
    return hash;
  }

  /// <summary>
  /// 等値比較演算子のフレンド宣言
  /// </summary>
  /// <param name="lhs">[in] 比較対象</param>
  /// <param name="rhs">[in] 比較対象</param>
  /// <returns>比較結果</returns>
  friend bool operator==(const DistanceTransform_& lhs, const DistanceTransform_& rhs) {
    return hash_value(lhs) == hash_value(rhs);
  }

  /// <summary>
  /// ストリーム出力関数のフレンド宣言
  /// </summary>
  /// <param name="os">[in] 出力ストリーム</param>
  /// <param name="rhs">[in] 出力オブジェクト</param>
  /// <returns>ストリーム</returns>
  template <typename charT, typename traits>
  friend std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os,
                                                       const DistanceTransform_& rhs) {
    os << hash_value(rhs);
    return os;
  }

  //! @name シリアライズ用設定
  //@{
  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned) {
    ar& boost::serialization::make_nvp("Table", m_table);
    ar& boost::serialization::make_nvp("Buffer", m_imgBuf);
  }
  //@}

#pragma endregion

 public:
#pragma region デフォルトメソッド定義

  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  DistanceTransform_() = default;

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  DistanceTransform_(const DistanceTransform_&) = default;

  /// <summary>
  /// 代入演算子
  /// </summary>
  DistanceTransform_& operator=(const DistanceTransform_&) = default;

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  DistanceTransform_(DistanceTransform_&&) = default;

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  DistanceTransform_& operator=(DistanceTransform_&&) = default;

  /// <summary>
  /// デストラクタ
  /// </summary>
  ~DistanceTransform_() = default;

#pragma endregion

  /// <summary>
  /// コンストラクタ
  /// </summary>
  /// <param name="size">処理画像サイズ</param>
  /// <remarks>処理の最適化のため、画像サイズを事前に指定しておかなければならない。</remarks>
  explicit DistanceTransform_(const cv::Size& size) { this->Init(size); }

  /// <summary>
  /// 初期化
  /// </summary>
  /// <param name="size">処理画像サイズ</param>
  /// <remarks>処理の最適化のため、画像サイズを事前に指定しておかなければならない。</remarks>
  void Init(const cv::Size& size) {
    m_table.Init(size);
    m_imgBuf.Init(size.height, size.width);
  }

  /// <summary>
  /// 距離変換
  /// </summary>
  /// <param name="src">[in] 入力画像</param>
  /// <param name="dst">[dst] 出力画像</param>
  /// <param name="binarization">[in] 入力画像を2値化する関数</param>
  /// <param name="conv">[in] 変換関数</param>
  /// <remarks>事前に Init を実行し、初期化を行っておかなければならない。</remarks>
  /// @tparam Ts 入力画像の型
  /// @tparam Td 出力画像の型
  /// @tparam Fb 2値化関数オブジェクトの型
  /// @tparam Fc 変換関数オブジェクト
  template <typename Ts, typename Td, typename Fb, typename Fc>
  void Calc(const cv::Mat& src, cv::Mat& dst, const Fb& binarization, const Fc& conv) {
    detail::Transform2D<FloatType, Alloc_, PartitionerType, Ts, Td, Fb, Fc>(
        m_table, src, dst, m_imgBuf.Mat(), this->NcMember<0>(), this->NcMember<1>(), binarization,
        conv);
  }

  /// <summary>
  /// 距離変換
  /// </summary>
  /// <param name="src">[in] 入力画像</param>
  /// <param name="dst">[dst] 出力画像</param>
  /// <param name="binarization">[in] 入力画像を2値化する関数</param>
  /// @tparam Ts 入力画像の型
  /// @tparam Td 出力画像の型
  /// @tparam F 2値化関数オブジェクトの型
  template <typename Ts, typename Td, typename F>
  void Calc(const cv::Mat& src, cv::Mat& dst, const F& binarization) {
    this->Calc<Ts, Td, F>(src, dst, binarization, [](const auto val) { return std::sqrt(val); });
  }

  /// <summary>
  /// 距離変換
  /// </summary>
  /// <param name="src">[in] 入力画像</param>
  /// <param name="dst">[dst] 出力画像</param>
  /// @tparam Ts 入力画像の型
  /// @tparam Td 出力画像の型
  template <typename Ts, typename Td>
  void Calc(const cv::Mat& src, cv::Mat& dst) {
    this->Calc<Ts, Td>(src, dst, [](const auto val) { return val != 0; });
  }

  /// <summary>
  /// 距離変換(2乗距離)
  /// </summary>
  /// <param name="src">[in] 入力画像</param>
  /// <param name="dst">[dst] 出力画像</param>
  /// <param name="binarization">[in] 入力画像を2値化する関数</param>
  /// <remarks>事前に Init を実行し、初期化を行っておかなければならない。</remarks>
  /// @tparam Ts 入力画像の型
  /// @tparam Td 出力画像の型
  /// @tparam F 2値化関数オブジェクトの型
  template <typename Ts, typename Td, typename F>
  void CalcSq(const cv::Mat& src, cv::Mat& dst, const F& binarization) {
    this->Calc<Ts, Td, F>(src, dst, binarization, [](auto val) { return val; });
  }

  /// <summary>
  /// 距離変換(2乗距離)
  /// </summary>
  /// <param name="src">[in] 入力画像</param>
  /// <param name="dst">[dst] 出力画像</param>
  /// <remarks>事前に Init を実行し、初期化を行っておかなければならない。</remarks>
  /// @tparam Ts 入力画像の型
  /// @tparam Td 出力画像の型
  template <typename Ts, typename Td>
  void CalcSq(const cv::Mat& src, cv::Mat& dst) {
    this->CalcSq<Ts, Td>(src, dst, [](const auto val) { return val != 0; });
  }
};

/// <summary>
/// 詳細実装
/// </summary>
namespace detail {

/// <summary>
/// 計算テーブル
/// </summary>
/// <remarks>インデックスの二乗などの事前計算可能な計算テーブル</remarks>
/// @tparam T 計算テーブルの型
/// @tparam Alloc_ テーブルのアロケータ
template <typename T, template <typename...> class Alloc_>
struct TableWithIndex_ {
  //! ベクトルタイプ
  //!@ U コンテナの中身の型
  template <typename U>
  using VecType = std::vector<U, Alloc_<U>>;
  //! テーブルサイズ
  int Size = 0;
  //! 二乗テーブル
  VecType<T> Sqr = {};
  //! 逆数テーブル
  VecType<T> Inv = {};
  //! 飽和分岐テーブルのスタートインデックス
  int SatStart = 0;

#pragma region デフォルトメソッド定義

  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  TableWithIndex_() = default;

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  TableWithIndex_(const TableWithIndex_&) = default;

  /// <summary>
  /// 代入演算子
  /// </summary>
  TableWithIndex_& operator=(const TableWithIndex_&) = default;

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  TableWithIndex_(TableWithIndex_&&) = default;

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  TableWithIndex_& operator=(TableWithIndex_&&) = default;

  /// <summary>
  /// デストラクタ
  /// </summary>
  ~TableWithIndex_() = default;

#pragma endregion

  /// <summary>
  /// 初期化
  /// </summary>
  /// <param name="size">テーブルサイズ</param>
  void Init(const cv::Size& size) {
    this->Size = std::max(size.width, size.height);

    this->Sqr.resize(this->Size << 1);
    this->Inv.resize(this->Size);
    SatStart = (this->Size << 1) + 1;

    auto* ptrSqr0 = &this->Sqr[0];
    auto* ptrSqr1 = ptrSqr0 + this->Size;
    auto* ptrInv = &this->Inv[0];
    for (auto i = 0; i < this->Size; ++i, ++ptrSqr0, ++ptrSqr1, ++ptrInv) {
      *ptrSqr0 = i * i;
      *ptrSqr1 = std::numeric_limits<T>::max();
      *ptrInv = T(0.5) / i;
    }
  }

 private:
#pragma region friend宣言

  /// <summary>
  /// ハッシュ関数のフレンド宣言
  /// </summary>
  /// <param name="rhs">[in] ハッシュを計算する対象</param>
  /// <returns>ハッシュ値</returns>
  /// <remarks>boost::hash でハッシュ値を取得出来るようにする。</remarks>
  friend std::size_t hash_value(const TableWithIndex_& rhs) {
    auto hash = boost::hash_value(rhs.Size);
    boost::hash_combine(hash, rhs.Sqr);
    boost::hash_combine(hash, rhs.Inv);
    boost::hash_combine(hash, rhs.SatStart);
    return hash;
  }

  /// <summary>
  /// 等値比較演算子のフレンド宣言
  /// </summary>
  /// <param name="lhs">[in] 比較対象</param>
  /// <param name="rhs">[in] 比較対象</param>
  /// <returns>比較結果</returns>
  friend bool operator==(const TableWithIndex_& lhs, const TableWithIndex_& rhs) {
    return hash_value(lhs) == hash_value(rhs);
  }

  /// <summary>
  /// 等値比較演算子のフレンド宣言
  /// </summary>
  /// <param name="lhs">[in] 比較対象</param>
  /// <param name="rhs">[in] 比較対象</param>
  /// <returns>比較結果</returns>
  friend bool operator!=(const TableWithIndex_& lhs, const TableWithIndex_& rhs) {
    return !(lhs == rhs);
  }

  /// <summary>
  /// ストリーム出力関数のフレンド宣言
  /// </summary>
  /// <param name="os">[in] 出力ストリーム</param>
  /// <param name="rhs">[in] 出力オブジェクト</param>
  /// <returns>ストリーム</returns>
  template <typename charT, typename traits>
  friend std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os,
                                                       const TableWithIndex_& rhs) {
    os << hash_value(rhs);
    return os;
  }

  //! @name シリアライズ用設定
  //@{
  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned) {
    ar& BOOST_SERIALIZATION_NVP(Size);
    ar& BOOST_SERIALIZATION_NVP(Sqr);
    ar& BOOST_SERIALIZATION_NVP(Inv);
    ar& BOOST_SERIALIZATION_NVP(SatStart);
  }
  //@}

#pragma endregion
};

/// <summary>
/// 単純な1次元の距離変換
/// </summary>
/// <param name="sqrTbl">[in] 二乗テーブル</param>
/// <param name="satTbl">[in] 飽和判定テーブル</param>
/// <param name="size">[in] サイズ</param>
/// <param name="byteStep">[in] ステップ(byte)</param>
/// <param name="s">[in] 入力データのポインタ</param>
/// <param name="ptrD0">[out] 出力データのポインタ</param>
/// <param name="ptrI0">[out] 最短インデックスのポインタ</param>
/// <param name="v">[out] インデックスバッファのポインタ</param>
/// <param name="binarization">[in] 二値化関数</param>
/// <remarks>
/// <para>初回のシンプルな距離計算</para>
/// <para>効率化のために、入力画像の二値化処理も同時に行う。</para>
/// </remarks>
/// @tparam T 出力・一時バッファの型
/// @tparam Ts 入力の型
/// @tparam F 入力を二値化する関数
template <typename T, typename Ts, typename F>
void TransformSimple1DwidthIndex(const T* const sqrTbl, const int size, const int byteStep,
                                 const Ts* const s, T* ptrD0, int* ptrI0, int* ptrV0,
                                 const F& binarization) {
  const auto sizeM1 = size - 1;
  const auto* ptrS = s + sizeM1;
  auto* ptrV = ptrV0 + sizeM1;
  auto* ptrI1 = ptrI0 + sizeM1;
  auto dist = sizeM1;
  for (auto i = sizeM1, idx = sizeM1; i >= 0; --i, --ptrS, --ptrV, --ptrI1) {
    if (binarization(*ptrS)) {
      ++dist;
    } else {
      dist = 0;
      idx = i;
    }
    *ptrV = dist;
    *ptrI1 = idx;
  }
  dist = sizeM1;
  for (auto i = 0, idx = *ptrI0; i < size;
       ++i, ++ptrV0, ptrD0 = cv::PtrByteInc(ptrD0, byteStep), ++ptrI0) {
    const auto diff = dist - *ptrV0;
    if (diff < 0) {
      ++dist;
    } else {
      dist -= diff;
      idx = *ptrI0;
    }
    *ptrV0 = dist;
    *ptrD0 = sqrTbl[dist];
    *ptrI0 = idx;
  }
}

/// <summary>
/// 単純な1次元の距離変換(並列実行対応)
/// </summary>
/// <param name="tbl">[in] 計算テーブル</param>
/// <param name="size">[in] サイズ</param>
/// <param name="byteStep">[in] ステップ(byte)</param>
/// <param name="s">[in] 入力データのポインタ</param>
/// <param name="ptrD0">[out] 出力データのポインタ</param>
/// <param name="ptrI0">[out] 最短インデックスのポインタ</param>
/// <param name="binarization">[in] 二値化関数</param>
/// <remarks>
/// <para>距離変換処理を並列実行するために、TLSバッファを使用する。</para>
/// <para>そのため、この関数は並列スレッドの中で呼び出さなければならない。</para>
/// </remarks>
/// @tparam T 出力・一時バッファの型
/// @tparam Alloc_ バッファのアロケータ
/// @tparam Ts 入力の型
/// @tparam F 入力を二値化する関数
template <typename T, template <typename...> class Alloc_, typename Ts, typename F>
void TransformSimple1DwidthIndex(const TableWithIndex_<T, Alloc_>& tbl, const int size,
                                 const int byteStep, const Ts* const s, T* ptrD0, int* ptrI0,
                                 const F& binarization) {
  auto buf = GetTlsBuffer<T, Alloc_>(tbl.Size);
  auto& bufIdx = std::get<0>(buf);
  TransformSimple1DwidthIndex<T, Ts, F>(&tbl.Sqr[0], size, byteStep, s, ptrD0, ptrI0, &bufIdx[0],
                                        binarization);
}

/// <summary>
/// 1次元の距離変換
/// </summary>
/// <param name="sqrTbl">[in] 二乗テーブル</param>
/// <param name="invTbl">[in] 逆数テーブル</param>
/// <param name="size">[in] サイズ</param>
/// <param name="byteStep">[in] ステップ(byte)</param>
/// <param name="f">[in] 入力データのポインタ</param>
/// <param name="ptrD0">[out] 出力データのポインタ</param>
/// <param name="ptrI0">[out] 最短インデックスのポインタ</param>
/// <param name="ptrV0">[out] インデックスバッファのポインタ</param>
/// <param name="ptrZ0">[out] 一時バッファのポインタ</param>
/// <param name="conv">[in] 出力変換関数</param>
/// <remarks>
/// <para>2回目の距離計算</para>
/// <para>効率化のために、L2距離から出力値への変換処理も同時に行う。</para>
/// </remarks>
/// @tparam T 入力・一時バッファの型
/// @tparam Td 出力の型
/// @tparam F L2距離を出力値に変換する関数
template <typename T, typename Td, typename F>
void Transform1DwidthIndex(const T* const sqrTbl, const T* const invTbl, const int size,
                           const int byteStep, const T* const f, Td* ptrD0, int* ptrI0, int* ptrV0,
                           T* ptrZ0, const F& conv) {
  const auto* ptrF1 = f + 1;
  auto* ptrV = ptrV0;
  auto* ptrZ1 = ptrZ0 + 1;

  *ptrV = 0;
  *ptrZ0 = std::numeric_limits<T>::lowest();
  *ptrZ1 = std::numeric_limits<T>::max();
  for (auto q = 1; q < size; ++q, ++ptrF1) {
    const auto A = (*ptrF1) + (q * q);
    const auto v1 = *ptrV;
    auto s = (A - (f[v1] + sqrTbl[v1])) * invTbl[q - v1];
    while (s <= *ptrZ0) {
      --ptrV;
      --ptrZ0;

      const auto v2 = *ptrV;
      s = (A - (f[v2] + sqrTbl[v2])) * invTbl[q - v2];
    }
    ++ptrV;
    ++ptrZ0;

    *ptrV = q;
    *ptrZ0 = s;
    *(ptrZ0 + 1) = std::numeric_limits<T>::max();
  }

  for (auto q = 0; q < size; ++q, ptrD0 = cv::PtrByteInc(ptrD0, byteStep), ++ptrI0) {
    while (*ptrZ1 < q) {
      ++ptrV0;
      ++ptrZ1;
    }
    const auto v0 = *ptrV0;
    *ptrD0 = cv::saturate_cast<Td>(conv(sqrTbl[std::abs(q - v0)] + f[v0]));
    *ptrI0 = v0;
  }
}

/// <summary>
/// 1次元の距離変換(並列実行対応)
/// </summary>
/// <param name="tbl">[in] 計算テーブル</param>
/// <param name="size">[in] サイズ</param>
/// <param name="byteStep">[in] ステップ(byte)</param>
/// <param name="f">[in] 入力データのポインタ</param>
/// <param name="ptrD0">[out] 出力データのポインタ</param>
/// <param name="ptrI0">[out] 最短インデックスのポインタ</param>
/// <param name="conv">[in] 出力変換関数</param>
/// <remarks>距離変換処理を並列実行するために、スレッド毎のバッファを用意する。</remarks>
/// @tparam T 入力・一時バッファの型
/// @tparam Alloc_ バッファのアロケータ
/// @tparam Td 出力の型
/// @tparam F L2距離を出力値に変換する関数
template <typename T, template <typename...> class Alloc_, typename Td, typename F>
void Transform1DwidthIndex(const TableWithIndex_<T, Alloc_>& tbl, const int size,
                           const int byteStep, const T* const f, T* ptrD0, int* ptrI0,
                           const F& conv) {
  auto buf = GetTlsBuffer<T, Alloc_>(tbl.Size);
  auto& bufIdx = std::get<0>(buf);
  auto& bufTmp = std::get<1>(buf);
  Transform1DwidthIndex<T>(&tbl.Sqr[0], &tbl.Inv[0], size, byteStep, f, ptrD0, ptrI0, &bufIdx[0],
                           &bufTmp[size], conv);
}

/// <summary>
/// 2次元の距離変換
/// </summary>
/// <param name="tbl">[in] 計算テーブル</param>
/// <param name="src">[in] 入力</param>
/// <param name="dst">[out] 出力</param>
/// <param name="idxX">[out] インデックスバッファ(x)</param>
/// <param name="idxY">[out] インデックスバッファ(y)</param>
/// <param name="bufSqDist">[out] 二乗距離計算用バッファ</param>
/// <param name="paX">[in] パーティショナ(x方向計算用)</param>
/// <param name="paY">[in] パーティショナ(y方向計算用)</param>
/// <param name="binarization">[in] 二値化関数</param>
/// <param name="conv">[in] 出力変換関数</param>
/// <remarks>
/// <para>距離変換アルゴリズムの特性上、1行または1列毎に距離計算を行う。</para>
/// <para>この実装では、先に列方向(x方向)で距離計算を行ってから行方向(y方向)の距離計算を行う。</para>
/// <para>高速化のために方向毎に処理を並列化するが、最初の並列化(x方向)の際には</para>
/// <para>入力画像の2値化処理、二回目の並列化(y方向)の際には、出力値変換処理を同時に行う。</para>
/// <para>また、距離計算の際に算出される最短インデックスを保持しておくことにより、</para>
/// <para>後から最短座標を計算できるようになっている。</para>
/// </remarks>
/// @tparam T 一時バッファの型
/// @tparam Alloc_ バッファのアロケータ
/// @tparam P パーティショナの型
/// @tparam Ts 入力画像の型
/// @tparam Td 出力画像の型
/// @tparam Fb 2値化関数オブジェクトの型
/// @tparam Fc L2距離を出力値に変換する関数
template <typename T, template <typename...> class Alloc_, typename P, typename Ts, typename Td,
          typename Fb, typename Fc>
void Transform2DwidthIndex(const TableWithIndex_<T, Alloc_>& tbl, const cv::Mat& src, cv::Mat& dst,
                           cv::Mat idxX, cv::Mat idxY, cv::Mat bufSqDist, P& paX, P& paY,
                           const Fb& binarization, const Fc& conv) {
  // 行単位で並列化
  tbb::tbb_for(0, src.rows,
               [tbl, size = src.cols, byteStep = bufSqDist.step[0], &src, &idxX, &bufSqDist,
                &binarization](const auto y) {
                 const auto* const ptrSrc = src.ptr<Ts>(y);
                 auto* ptrBufSqDist = bufSqDist.ptr<T>(0, y);
                 auto* ptrIdxX = idxX.ptr<int>(y);
                 TransformSimple1DwidthIndex<T, Alloc_, Ts, Fb>(
                     tbl, size, byteStep, ptrSrc, ptrBufSqDist, ptrIdxX, binarization);
               },
               paX);

  // 列単位で並列化
  tbb::tbb_for(
      0, dst.cols,
      [tbl, size = dst.rows, byteStep = dst.step[0], &dst, &idxY, &bufSqDist, &conv](const auto x) {
        const auto* const ptrBufSqDist = bufSqDist.ptr<T>(x);
        auto* ptrDst = dst.ptr<T>(0, x);
        auto* ptrIdxY = idxY.ptr<int>(x);
        Transform1DwidthIndex<T, Alloc_, Td, Fc>(tbl, size, byteStep, ptrBufSqDist, ptrDst, ptrIdxY,
                                                 conv);
      },
      paY);
}

}  // namespace detail

/// <summary>
/// 距離変換画像計算クラス
/// </summary>
/// <remarks>
/// <para>アルゴリズム詳細は以下参照</para>
/// <para>Pedro Felzenszwalb and Daniel Huttenlocher. Distance transforms of sampled functions.
/// Technical report, Cornell University, 2004.</para>
/// </remarks>
/// @tparam Tf 内部計算の型
/// @tparam P パーティショナの型
/// @tparam Alloc_ バッファのアロケータ
template <typename Tf = float, typename P = tbb::affinity_partitioner,
          template <typename...> class Alloc_ = tbb::cache_aligned_allocator>
class DistanceTransformWithIndex_
    : public commonutility::NonCopyMembers_<P, P>,
      boost::equality_comparable<DistanceTransformWithIndex_<Tf, P, Alloc_>,
                                 DistanceTransformWithIndex_<Tf, P, Alloc_>> {
  static_assert(std::is_floating_point<Tf>{}, "template parameter Tf must be floatng point type");

 public:
  //! 浮動小数点タイプ
  using FloatType = Tf;
  //! パーティショナタイプ
  using PartitionerType = P;

  //! バッファの種別定義
  enum class Buf { X, Y, Buf };
  //! バッファ定義クラス
  //@tparam I 種別
  //@tparam V 値の型
  //@tparam C チャンネル
  //@tparam D ダンプフラグ
  template <Buf I, typename V, int C = 1, bool D = false>
  using I_ = commonutility::ImageSetting_<Buf, I, V, C, D>;
  //! バッファタイプ
  using BufType = commonutility::ImageCollectionImpl_<Alloc_, I_<Buf::X, int>, I_<Buf::Y, int>,
                                                      I_<Buf::Buf, FloatType>>;

 private:
  //! テーブルタイプ
  using TableType = detail::TableWithIndex_<FloatType, Alloc_>;

  //! 計算テーブル
  TableType m_table = {};
  //! 計算バッファ
  BufType m_imgBuf = {};

#pragma region friend宣言

  /// <summary>
  /// ハッシュ関数のフレンド宣言
  /// </summary>
  /// <param name="rhs">[in] ハッシュを計算する対象</param>
  /// <returns>ハッシュ値</returns>
  /// <remarks>boost::hash でハッシュ値を取得出来るようにする。</remarks>
  friend std::size_t hash_value(const DistanceTransformWithIndex_& rhs) {
    auto hash = hash_value(rhs.m_table);
    boost::hash_combine(hash, rhs.m_imgBuf);
    return hash;
  }

  /// <summary>
  /// 等値比較演算子のフレンド宣言
  /// </summary>
  /// <param name="lhs">[in] 比較対象</param>
  /// <param name="rhs">[in] 比較対象</param>
  /// <returns>比較結果</returns>
  friend bool operator==(const DistanceTransformWithIndex_& lhs,
                         const DistanceTransformWithIndex_& rhs) {
    return hash_value(lhs) == hash_value(rhs);
  }

  /// <summary>
  /// ストリーム出力関数のフレンド宣言
  /// </summary>
  /// <param name="os">[in] 出力ストリーム</param>
  /// <param name="rhs">[in] 出力オブジェクト</param>
  /// <returns>ストリーム</returns>
  template <typename charT, typename traits>
  friend std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os,
                                                       const DistanceTransformWithIndex_& rhs) {
    os << hash_value(rhs);
    return os;
  }

  //! @name シリアライズ用設定
  //@{
  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned) {
    ar& boost::serialization::make_nvp("Table", m_table);
    ar& boost::serialization::make_nvp("Buffer", m_imgBuf);
  }
  //@}

#pragma endregion

 public:
#pragma region デフォルトメソッド定義

  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  DistanceTransformWithIndex_() = default;

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  DistanceTransformWithIndex_(const DistanceTransformWithIndex_&) = default;

  /// <summary>
  /// 代入演算子
  /// </summary>
  DistanceTransformWithIndex_& operator=(const DistanceTransformWithIndex_&) = default;

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  DistanceTransformWithIndex_(DistanceTransformWithIndex_&&) = default;

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  DistanceTransformWithIndex_& operator=(DistanceTransformWithIndex_&&) = default;

  /// <summary>
  /// デストラクタ
  /// </summary>
  ~DistanceTransformWithIndex_() = default;

#pragma endregion

  /// <summary>
  /// コンストラクタ
  /// </summary>
  /// <param name="size">処理画像サイズ</param>
  /// <remarks>処理の最適化のため、画像サイズを事前に指定しておかなければならない。</remarks>
  explicit DistanceTransformWithIndex_(const cv::Size& size) { this->Init(size); }

  /// <summary>
  /// 初期化
  /// </summary>
  /// <param name="size">処理画像サイズ</param>
  /// <remarks>処理の最適化のため、画像サイズを事前に指定しておかなければならない。</remarks>
  void Init(const cv::Size& size) {
    m_table.Init(size);
    m_imgBuf.Init<Buf::X>(size);
    m_imgBuf.Init<Buf::Y, Buf::Buf>(size.height, size.width);
  }

  /// <summary>
  /// 距離変換
  /// </summary>
  /// <param name="src">[in] 入力画像</param>
  /// <param name="dst">[dst] 出力画像</param>
  /// <param name="binarization">[in] 入力画像を2値化する関数</param>
  /// <param name="conv">[in] 変換関数</param>
  /// <remarks>事前に Init を実行し、初期化を行っておかなければならない。</remarks>
  /// @tparam Ts 入力画像の型
  /// @tparam Td 出力画像の型
  /// @tparam Fb 2値化関数オブジェクトの型
  /// @tparam Fc 変換関数オブジェクト
  template <typename Ts, typename Td, typename Fb, typename Fc>
  void Calc(const cv::Mat& src, cv::Mat& dst, const Fb& binarization, const Fc& conv) {
    detail::Transform2DwidthIndex<FloatType, Alloc_, PartitionerType, Ts, Td, Fb, Fc>(
        m_table, src, dst, m_imgBuf.Mat<Buf::X>(), m_imgBuf.Mat<Buf::Y>(), m_imgBuf.Mat<Buf::Buf>(),
        this->NcMember<0>(), this->NcMember<1>(), binarization, conv);
  }

  /// <summary>
  /// 距離変換
  /// </summary>
  /// <param name="src">[in] 入力画像</param>
  /// <param name="dst">[dst] 出力画像</param>
  /// <param name="binarization">[in] 入力画像を2値化する関数</param>
  /// @tparam Ts 入力画像の型
  /// @tparam Td 出力画像の型
  /// @tparam F 2値化関数オブジェクトの型
  template <typename Ts, typename Td, typename F>
  void Calc(const cv::Mat& src, cv::Mat& dst, const F& binarization) {
    this->Calc<Ts, Td, F>(src, dst, binarization, [](const auto val) { return std::sqrt(val); });
  }

  /// <summary>
  /// 距離変換
  /// </summary>
  /// <param name="src">[in] 入力画像</param>
  /// <param name="dst">[dst] 出力画像</param>
  /// @tparam Ts 入力画像の型
  /// @tparam Td 出力画像の型
  template <typename Ts, typename Td>
  void Calc(const cv::Mat& src, cv::Mat& dst) {
    this->Calc<Ts, Td>(src, dst, [](const auto val) { return val != 0; });
  }

  /// <summary>
  /// 距離変換(2乗距離)
  /// </summary>
  /// <param name="src">[in] 入力画像</param>
  /// <param name="dst">[dst] 出力画像</param>
  /// <param name="binarization">[in] 入力画像を2値化する関数</param>
  /// <remarks>事前に Init を実行し、初期化を行っておかなければならない。</remarks>
  /// @tparam Ts 入力画像の型
  /// @tparam Td 出力画像の型
  /// @tparam F 2値化関数オブジェクトの型
  template <typename Ts, typename Td, typename F>
  void CalcSq(const cv::Mat& src, cv::Mat& dst, const F& binarization) {
    this->Calc<Ts, Td, F>(src, dst, binarization, [](auto val) { return val; });
  }

  /// <summary>
  /// 距離変換(2乗距離)
  /// </summary>
  /// <param name="src">[in] 入力画像</param>
  /// <param name="dst">[dst] 出力画像</param>
  /// <remarks>事前に Init を実行し、初期化を行っておかなければならない。</remarks>
  /// @tparam Ts 入力画像の型
  /// @tparam Td 出力画像の型
  template <typename Ts, typename Td>
  void CalcSq(const cv::Mat& src, cv::Mat& dst) {
    this->CalcSq<Ts, Td>(src, dst, [](const auto val) { return val != 0; });
  }

  /// <summary>
  /// 最も近い座標を計算
  /// </summary>
  /// <param name="x">[in] 調べる座標(x)</param>
  /// <param name="y">[in] 調べる座標(y)</param>
  /// <returns>指定した座標から最も近い座標</returns>
  cv::Point CalcPoint(const int x, const int y) {
    const auto y_ = *m_imgBuf.Mat<Buf::Y>().ptr<BufType::ValueType_<Buf::Y>>(x, y);
    return {*m_imgBuf.Mat<Buf::X>().ptr<BufType::ValueType_<Buf::X>>(y_, x), y_};
  }

  /// <summary>
  /// 最も近い座標を計算
  /// </summary>
  /// <param name="pos">[in] 調べる座標</param>
  /// <returns>指定した座標から最も近い座標</returns>
  cv::Point CalcPoint(const cv::Point& pos) { return this->CalcPoint(pos.x, pos.y); }

  /// <summary>
  /// 最も近い座標を計算
  /// </summary>
  /// <param name="points">[out] 座標マップ</param>
  void CalcPoint(cv::Mat& points) {
    using XType = BufType::ValueType_<Buf::X>;
    using XType = BufType::ValueType_<Buf::Y>;
    tbb::tbb_for(points.size(), [&points, idxX = m_imgBuf.Mat<Buf::X>(),
                                 idxY = m_imgBuf.Mat<Buf::Y>()](const auto& range) {
      const auto stepX = idxX.step[0];
      const auto stepY = idxY.step[0];

      const auto& rows = range.rows();
      const auto& cols = range.cols();
      auto y = std::begin(rows);
      for (const auto ye = std::end(rows); y < ye; ++y) {
        auto x = std::begin(cols);
        const auto* ptrX = idxY.ptr<BufType::ValueType_<Buf::X>>(0, x);
        const auto* ptrY = idxY.ptr<BufType::ValueType_<Buf::Y>>(x, y);
        auto* ptrPoint = points.ptr<cv::Point>(y, x);
        for (const auto xe = std::end(cols); x < xe;
             ++x, ++ptrX, ptrY = cv::PtrByteInc(ptrY, stepY), ++ptrPoint) {
          const auto& y_ = *ptrY;
          ptrPoint->x = *cv::PtrByteInc(ptrX, y_ * stepX);
          ptrPoint->y = y_;
        }
      }
    });
  }

  /// <summary>
  /// 画像バッファ取得
  /// </summary>
  /// <returns>画像バッファ</returns>
  const BufType& GetImageBuffer() const { return m_imgBuf; }
};

}  // namespace imgproc

#pragma once

#include "CommonDef.h"
#include "OpenCvConfig.h"
#include "TbbConfig.h"

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <cmath>

#include <algorithm>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_member.hpp>

#include <opencv2/core.hpp>

// 警告抑制解除
MSVC_WARNING_POP

/// <summary>
/// 画像処理
/// </summary>
namespace imgproc { namespace old {

/// <summary>
/// 距離変換
/// </summary>
template <typename T = double, typename P = tbb::auto_partitioner>
class DistanceTransform_ {
 public:
  using PartitionerType = P;

 private:
  //! バッファは浮動小数点型のみ
  static_assert(std::is_floating_point<T>{}, "template parameter T must be floatng point type");

  //
  // メンバ変数
  //

  //! 画像のサイズ
  cv::Size m_size = {};
  //! 計算結果(2乗距離)
  std::vector<T> m_sqDist = {};
  //! インデックス(X)
  std::vector<int> m_idxX = {};
  //! インデックス(Y)
  std::vector<int> m_idxY = {};
  //! 二乗距離バッファ
  std::vector<T> m_bufSqDist = {};
  //! 計算用バッファ
  std::vector<int> m_bufIdx = {};
  //! 計算用バッファ
  std::vector<T> m_bufZ = {};

  //! パーティショナー群
  struct Partitioners {
    PartitionerType Pa = {};
    PartitionerType X = {};
    PartitionerType Y = {};
  };
  //! パーティショナー群
  Partitioners m_partitioners = {};

  //! @name シリアライズ用設定
  //@{
  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();
  template <typename Archive>
  void save(Archive& ar, const unsigned) const {
    ar& boost::serialization::make_nvp("Size", m_size);
    ar& boost::serialization::make_nvp("SqDist", m_sqDist);
    ar& boost::serialization::make_nvp("IdxX", m_idxX);
    ar& boost::serialization::make_nvp("IdxY", m_idxY);
  }
  template <typename Archive>
  void load(Archive& ar, const unsigned) {
    ar& boost::serialization::make_nvp("Size", m_size);
    ar& boost::serialization::make_nvp("SqDist", m_sqDist);
    ar& boost::serialization::make_nvp("IdxX", m_idxX);
    ar& boost::serialization::make_nvp("IdxY", m_idxY);

    const size_t bufSize = m_size.area();
    m_bufSqDist.resize(bufSize);
    m_bufIdx.resize(bufSize);
    m_bufZ.resize(bufSize + std::max(m_size.width, m_size.height));
  }
  //@}

  /// <summary>
  /// 1次元の距離変換
  /// </summary>
  /// <param name="size">[in] サイズ</param>
  /// <param name="step">[in] ステップ</param>
  /// <param name="f">[in] 入力データ</param>
  /// <param name="d">[out] バッファ</param>
  /// <param name="i">[out] バッファ</param>
  /// <param name="v">[out] バッファ</param>
  /// <param name="z">[out] バッファ</param>
  template <typename Tb>
  static void Transform1D(const int size, const int step, const Tb* const f, Tb* const d,
                          int* const i, int* const v, Tb* const z) {
    const Tb* ptrF1 = f + 1;
    int* ptrV0 = v;
    Tb* ptrZ0 = z;
    Tb* ptrZ1 = z + 1;

    *ptrV0 = 0;
    *ptrZ0 = std::numeric_limits<Tb>::lowest();
    *ptrZ1 = std::numeric_limits<Tb>::max();
    for (int q = 1; q < size; ++q, ++ptrF1) {
      const Tb A = (*ptrF1) + (q * q);
      const int& v1 = *ptrV0;
      Tb s = (A - (f[v1] + (v1 * v1))) / ((q - v1) << 1);
      while (s <= *ptrZ0) {
        --ptrV0;
        --ptrZ0;

        const int& v2 = *ptrV0;
        s = (A - (f[v2] + (v2 * v2))) / ((q - v2) << 1);
      }
      ++ptrV0;
      ++ptrZ0;

      *ptrV0 = q;
      *ptrZ0 = s;
      *(ptrZ0 + 1) = std::numeric_limits<Tb>::max();
    }

    Tb* ptrD0 = d;
    int* ptrI0 = i;
    ptrV0 = v;
    for (int q = 0; q < size; ++q, ptrD0 += step, ++ptrI0) {
      while (*ptrZ1 < q) {
        ++ptrV0;
        ++ptrZ1;
      }
      const int& refV = *ptrV0;
      const auto qv = q - refV;
      *ptrD0 = qv * qv + f[refV];
      *ptrI0 = refV;
    }
  }

  /// <summary>
  /// 2次元の距離変換
  /// </summary>
  /// <param name="size">[int] 画像サイズ</param>
  /// <param name="data">[int,out] 変換前後のデータ</param>
  /// <param name="idxX">[out] 最短座標(X)</param>
  /// <param name="idxY">[out] 最短座標(Y)</param>
  /// <param name="work">[out] バッファ</param>
  /// <param name="workV">[out] バッファ</param>
  /// <param name="workZ">[out] バッファ</param>
  template <typename Tb>
  static void Transform2D(const cv::Size& size, std::vector<Tb>& data, std::vector<int>& idxX,
                          std::vector<int>& idxY, std::vector<Tb>& work, std::vector<int>& workV,
                          std::vector<Tb>& workZ, PartitionerType& paX, PartitionerType& paY) {
    // 横方向の距離変換
    tbb::tbb_for(0, size.height,
                 [&](const int y) {
                   const int idx = y * size.width;

                   // ※vs2010のラムダ式の不具合により、template引数も記述
                   DistanceTransform_<T, P>::Transform1D(size.width, size.height, &data[idx],
                                                         &work[y], &idxX[idx], &workV[idx],
                                                         &workZ[idx + y]);
                 },
                 paX);

    // 縦方向の距離変換
    tbb::tbb_for(0, size.width,
                 [&](const int x) {
                   const int idx = x * size.height;

                   // ※vs2010のラムダ式の不具合により、template引数も記述
                   DistanceTransform_<T, P>::Transform1D(size.height, size.width, &work[idx],
                                                         &data[x], &idxY[idx], &workV[idx],
                                                         &workZ[idx + x]);
                 },
                 paY);
  }

  /// <summary>
  /// 二値化&距離変換(二乗距離)
  /// </summary>
  /// <param name="src">[in] 入力</param>
  /// <param name="comp">[in] 二値化関数</param>
  /// <returns>変換結果</returns>
  /// <remarks>平方根をしてない結果を返すメソッド</remarks>
  template <typename Ts, typename F>
  const cv::Mat TransformSqDist(const cv::Mat& src, const F& comp) {
    Assert(src.channels() == 1 && src.size() == m_size, "Src-Data Error");

    // 2値データを浮動小数点型に変換
    cv::Mat buf(m_size, cv::Type<T>(), &m_sqDist[0]);
    tbb::tbb_for(src.size(),
                 [&](const tbb::blocked_range2d<int>& range) {
                   for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
                     int x = std::begin(range.cols());
                     const Ts* ptrSrc = src.ptr<Ts>(y, x);
                     T* ptrBuf = buf.ptr<T>(y, x);
                     for (; x < std::end(range.cols()); ++x, ++ptrSrc, ++ptrBuf)
                       *ptrBuf = comp(*ptrSrc) ? std::numeric_limits<T>::max() : T(0);
                   }
                 },
                 m_partitioners.Pa);

    // 変換
    DistanceTransform_::Transform2D(m_size, m_sqDist, m_idxX, m_idxY, m_bufSqDist, m_bufIdx, m_bufZ,
                                    m_partitioners.X, m_partitioners.Y);

    return buf;
  }

  /// <summary>
  /// 出力データ変換
  /// </summary>
  /// <param name="sqDist">[in] 二乗距離</param>
  /// <param name="dst">[out] 出力</param>
  /// <param name="conv">[in] 変換関数</param>
  /// <remarks>計算結果を出力型に変換する</remarks>
  template <typename Td, typename F>
  void ConvertTo(const cv::Mat& sqDist, cv::Mat& dst, const F& conv) {
    Assert(dst.channels() == 1 && dst.size() == m_size, "Dst-Data Error");

    // 出力データに変換
    tbb::tbb_for(dst.size(),
                 [&](const tbb::blocked_range2d<int>& range) {
                   for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
                     int x = std::begin(range.cols());
                     const T* ptrSqDist = sqDist.ptr<T>(y, x);
                     Td* ptrDst = dst.ptr<Td>(y, x);
                     for (; x < std::end(range.cols()); ++x, ++ptrSqDist, ++ptrDst)
                       *ptrDst = conv(*ptrSqDist);
                   }
                 },
                 m_partitioners.Pa);
  }

 public:
  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  DistanceTransform_() = default;

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  DistanceTransform_(const DistanceTransform_& rhs)
      : m_size(rhs.m_size),
        m_sqDist(rhs.m_sqDist),
        m_idxX(rhs.m_idxX),
        m_idxY(rhs.m_idxY),
        m_bufSqDist(rhs.m_bufSqDist),
        m_bufIdx(rhs.m_bufIdx),
        m_bufZ(rhs.m_bufZ) {}

  void swap(DistanceTransform_& rhs) {
    std::swap(m_size, rhs.m_size);
    std::swap(m_sqDist, rhs.m_sqDist);
    std::swap(m_idxX, rhs.m_idxX);
    std::swap(m_idxY, rhs.m_idxY);
    std::swap(m_bufSqDist, rhs.m_bufSqDist);
    std::swap(m_bufIdx, rhs.m_bufIdx);
    std::swap(m_bufZ, rhs.m_bufZ);
  }

  /// <summary>
  /// 代入演算子
  /// </summary>
  DistanceTransform_& operator=(const DistanceTransform_& rhs) {
    DistanceTransform_ tmp = rhs;
    this->swap(tmp);
    return *this;
  }

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  DistanceTransform_(DistanceTransform_&&) = default;

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  DistanceTransform_& operator=(DistanceTransform_&&) = default;

  /// <summary>
  /// デストラクタ
  /// </summary>
  virtual ~DistanceTransform_() = default;

  // コンストラクタ
  explicit DistanceTransform_(const cv::Size& size)
      : m_size(size),
        m_sqDist(m_size.area()),
        m_idxX(m_size.area()),
        m_idxY(m_size.area()),
        m_bufSqDist(m_size.area()),
        m_bufIdx(m_size.area()),
        m_bufZ(m_size.area() + std::max(m_size.width, m_size.height)) {}

  // 初期化
  void Init(const cv::Size& size) {
    m_size = size;

    const size_t bufSize = m_size.area();
    m_sqDist.resize(bufSize);
    m_idxX.resize(bufSize);
    m_idxY.resize(bufSize);
    m_bufSqDist.resize(bufSize);
    m_bufIdx.resize(bufSize);
    m_bufZ.resize(bufSize + std::max(m_size.width, m_size.height));
  }

  // バッファサイズ取得
  const cv::Size& GetSize() const { return m_size; }

  //
  // ベースメソッド
  //

  // 距離変換
  template <typename Ts, typename Td>
  void CalcCompCast(const cv::Mat& src, cv::Mat& dst,
                    const std::function<const bool(const Ts&)>& comp,
                    const std::function<const Td(const T&)>& cast) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src, comp), dst, cast);
  }

  // 距離変換
  template <typename Ts, typename F>
  void CalcCompCast(const cv::Mat& src, cv::Mat& dst,
                    const std::function<const bool(const Ts&)>& comp, const F& cast) {
    switch (CV_MAT_DEPTH(dst.type())) {
      case CV_8U:
        this->CalcCompCast<Ts, uchar>(src, dst, comp, cast);
        break;
      case CV_8S:
        this->CalcCompCast<Ts, char>(src, dst, comp, cast);
        break;
      case CV_16U:
        this->CalcCompCast<Ts, ushort>(src, dst, comp, cast);
        break;
      case CV_16S:
        this->CalcCompCast<Ts, short>(src, dst, comp, cast);
        break;
      case CV_32S:
        this->CalcCompCast<Ts, int>(src, dst, comp, cast);
        break;
      case CV_32F:
        this->CalcCompCast<Ts, float>(src, dst, comp, cast);
        break;
      case CV_64F:
        this->CalcCompCast<Ts, double>(src, dst, comp, cast);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 距離変換
  template <typename F1, typename F2>
  void CalcCompCast(const cv::Mat& src, cv::Mat& dst, const F1& comp, const F2& cast) {
    switch (CV_MAT_DEPTH(src.type())) {
      case CV_8U:
        this->CalcCompCast<uchar>(src, dst, comp, cast);
        break;
      case CV_8S:
        this->CalcCompCast<char>(src, dst, comp, cast);
        break;
      case CV_16U:
        this->CalcCompCast<ushort>(src, dst, comp, cast);
        break;
      case CV_16S:
        this->CalcCompCast<short>(src, dst, comp, cast);
        break;
      case CV_32S:
        this->CalcCompCast<int>(src, dst, comp, cast);
        break;
      case CV_32F:
        this->CalcCompCast<float>(src, dst, comp, cast);
        break;
      case CV_64F:
        this->CalcCompCast<double>(src, dst, comp, cast);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  //
  // キャスト方法指定
  //

  // 距離変換(0以外)
  template <typename Ts, typename Td>
  void CalcCast(const cv::Mat& src, cv::Mat& dst, const std::function<const Td(const T&)>& cast) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src,
                                                  [](const Ts& val) -> bool {
                                                    return std::abs(val) >
                                                           std::numeric_limits<Ts>::epsilon();
                                                  }),
                        dst, cast);
  }

  // 距離変換(0のみ)
  template <typename Ts, typename Td>
  void CalcInvCast(const cv::Mat& src, cv::Mat& dst,
                   const std::function<const Td(const T&)>& cast) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src,
                                                  [](const Ts& val) -> bool {
                                                    return std::abs(val) <=
                                                           std::numeric_limits<Ts>::epsilon();
                                                  }),
                        dst, cast);
  }

  // 距離変換(0以外)
  template <typename Ts, typename F>
  void CalcCast(const cv::Mat& src, cv::Mat& dst, const F& cast) {
    switch (CV_MAT_DEPTH(dst.type())) {
      case CV_8U:
        this->CalcCast<Ts, uchar>(src, dst, cast);
        break;
      case CV_8S:
        this->CalcCast<Ts, char>(src, dst, cast);
        break;
      case CV_16U:
        this->CalcCast<Ts, ushort>(src, dst, cast);
        break;
      case CV_16S:
        this->CalcCast<Ts, short>(src, dst, cast);
        break;
      case CV_32S:
        this->CalcCast<Ts, int>(src, dst, cast);
        break;
      case CV_32F:
        this->CalcCast<Ts, float>(src, dst, cast);
        break;
      case CV_64F:
        this->CalcCast<Ts, double>(src, dst, cast);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 距離変換(0のみ)
  template <typename Ts, typename F>
  void CalcInvCast(const cv::Mat& src, cv::Mat& dst, const F& cast) {
    switch (CV_MAT_DEPTH(dst.type())) {
      case CV_8U:
        this->CalcInvCast<Ts, uchar>(src, dst, cast);
        break;
      case CV_8S:
        this->CalcInvCast<Ts, char>(src, dst, cast);
        break;
      case CV_16U:
        this->CalcInvCast<Ts, ushort>(src, dst, cast);
        break;
      case CV_16S:
        this->CalcInvCast<Ts, short>(src, dst, cast);
        break;
      case CV_32S:
        this->CalcInvCast<Ts, int>(src, dst, cast);
        break;
      case CV_32F:
        this->CalcInvCast<Ts, float>(src, dst, cast);
        break;
      case CV_64F:
        this->CalcInvCast<Ts, double>(src, dst, cast);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 距離変換(0以外)
  template <typename F>
  void CalcCast(const cv::Mat& src, cv::Mat& dst, const F& cast) {
    switch (CV_MAT_DEPTH(src.type())) {
      case CV_8U:
        this->CalcCast<uchar>(src, dst, cast);
        break;
      case CV_8S:
        this->CalcCast<char>(src, dst, cast);
        break;
      case CV_16U:
        this->CalcCast<ushort>(src, dst, cast);
        break;
      case CV_16S:
        this->CalcCast<short>(src, dst, cast);
        break;
      case CV_32S:
        this->CalcCast<int>(src, dst, cast);
        break;
      case CV_32F:
        this->CalcCast<float>(src, dst, cast);
        break;
      case CV_64F:
        this->CalcCast<double>(src, dst, cast);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 距離変換(0のみ)
  template <typename F>
  void CalcInvCast(const cv::Mat& src, cv::Mat& dst, const F& cast) {
    switch (CV_MAT_DEPTH(src.type())) {
      case CV_8U:
        this->CalcInvCast<uchar>(src, dst, cast);
        break;
      case CV_8S:
        this->CalcInvCast<char>(src, dst, cast);
        break;
      case CV_16U:
        this->CalcInvCast<ushort>(src, dst, cast);
        break;
      case CV_16S:
        this->CalcInvCast<short>(src, dst, cast);
        break;
      case CV_32S:
        this->CalcInvCast<int>(src, dst, cast);
        break;
      case CV_32F:
        this->CalcInvCast<float>(src, dst, cast);
        break;
      case CV_64F:
        this->CalcInvCast<double>(src, dst, cast);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  //
  // 比較方法指定
  //

  // 距離変換
  template <typename Ts, typename Td,
            std::enable_if_t<std::is_same<T, Td>{}, std::nullptr_t> = nullptr>
  void CalcComp(const cv::Mat& src, cv::Mat& dst,
                const std::function<const bool(const Ts&)>& comp) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src, comp), dst,
                        [](const T& val) -> Td { return std::sqrt(val); });
  }

  // 距離変換
  template <typename Ts, typename Td,
            std::enable_if_t<!std::is_same<T, Td>{}, std::nullptr_t> = nullptr>
  void CalcComp(const cv::Mat& src, cv::Mat& dst,
                const std::function<const bool(const Ts&)>& comp) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src, comp), dst,
                        [](const T& val) -> Td { return cv::saturate_cast<Td>(std::sqrt(val)); });
  }

  // 距離変換(距離の二乗)
  template <typename Ts, typename Td,
            std::enable_if_t<std::is_same<T, Td>{}, std::nullptr_t> = nullptr>
  void CalcCompSq(const cv::Mat& src, cv::Mat& dst,
                  const std::function<const bool(const Ts&)>& comp) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src, comp), dst,
                        [](const T& val) -> Td { return val; });
  }

  // 距離変換(距離の二乗)
  template <typename Ts, typename Td,
            std::enable_if_t<!std::is_same<T, Td>{}, std::nullptr_t> = nullptr>
  void CalcCompSq(const cv::Mat& src, cv::Mat& dst,
                  const std::function<const bool(const Ts&)>& comp) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src, comp), dst,
                        [](const T& val) -> Td { return cv::saturate_cast<Td>(val); });
  }

  // 距離変換
  template <typename Ts>
  void CalcComp(const cv::Mat& src, cv::Mat& dst,
                const std::function<const bool(const Ts&)>& comp) {
    switch (CV_MAT_DEPTH(dst.type())) {
      case CV_8U:
        this->CalcComp<Ts, uchar>(src, dst, comp);
        break;
      case CV_8S:
        this->CalcComp<Ts, char>(src, dst, comp);
        break;
      case CV_16U:
        this->CalcComp<Ts, ushort>(src, dst, comp);
        break;
      case CV_16S:
        this->CalcComp<Ts, short>(src, dst, comp);
        break;
      case CV_32S:
        this->CalcComp<Ts, int>(src, dst, comp);
        break;
      case CV_32F:
        this->CalcComp<Ts, float>(src, dst, comp);
        break;
      case CV_64F:
        this->CalcComp<Ts, double>(src, dst, comp);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 距離変換(距離の二乗))
  template <typename Ts>
  void CalcSq(const cv::Mat& src, cv::Mat& dst, const std::function<const bool(const Ts&)>& comp) {
    switch (CV_MAT_DEPTH(dst.type())) {
      case CV_8U:
        this->CalcCompSq<Ts, uchar>(src, dst, comp);
        break;
      case CV_8S:
        this->CalcCompSq<Ts, char>(src, dst, comp);
        break;
      case CV_16U:
        this->CalcCompSq<Ts, ushort>(src, dst, comp);
        break;
      case CV_16S:
        this->CalcCompSq<Ts, short>(src, dst, comp);
        break;
      case CV_32S:
        this->CalcCompSq<Ts, int>(src, dst, comp);
        break;
      case CV_32F:
        this->CalcCompSq<Ts, float>(src, dst, comp);
        break;
      case CV_64F:
        this->CalcCompSq<Ts, double>(src, dst, comp);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 距離変換
  template <typename F>
  void CalcComp(const cv::Mat& src, cv::Mat& dst, const F& comp) {
    switch (CV_MAT_DEPTH(src.type())) {
      case CV_8U:
        this->CalcComp<uchar>(src, dst, comp);
        break;
      case CV_8S:
        this->CalcComp<char>(src, dst, comp);
        break;
      case CV_16U:
        this->CalcComp<ushort>(src, dst, comp);
        break;
      case CV_16S:
        this->CalcComp<short>(src, dst, comp);
        break;
      case CV_32S:
        this->CalcComp<int>(src, dst, comp);
        break;
      case CV_32F:
        this->CalcComp<float>(src, dst, comp);
        break;
      case CV_64F:
        this->CalcComp<double>(src, dst, comp);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 距離変換(距離の二乗))
  template <typename F>
  void CalcCompSq(const cv::Mat& src, cv::Mat& dst, const F& comp) {
    switch (CV_MAT_DEPTH(src.type())) {
      case CV_8U:
        this->CalcCompSq<uchar>(src, dst, comp);
        break;
      case CV_8S:
        this->CalcCompSq<char>(src, dst, comp);
        break;
      case CV_16U:
        this->CalcCompSq<ushort>(src, dst, comp);
        break;
      case CV_16S:
        this->CalcCompSq<short>(src, dst, comp);
        break;
      case CV_32S:
        this->CalcCompSq<int>(src, dst, comp);
        break;
      case CV_32F:
        this->CalcCompSq<float>(src, dst, comp);
        break;
      case CV_64F:
        this->CalcCompSq<double>(src, dst, comp);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  //
  // キャスト方法、比較方法指定なし
  //

  // 距離変換(0以外)
  template <typename Ts, typename Td,
            std::enable_if_t<std::is_same<T, Td>{}, std::nullptr_t> = nullptr>
  void Calc(const cv::Mat& src, cv::Mat& dst) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src,
                                                  [](const Ts& val) -> bool {
                                                    return std::abs(val) >
                                                           std::numeric_limits<Ts>::epsilon();
                                                  }),
                        dst, [](const T& val) -> Td { return std::sqrt(val); });
  }

  // 距離変換(0以外)
  template <typename Ts, typename Td,
            std::enable_if_t<!std::is_same<T, Td>{}, std::nullptr_t> = nullptr>
  void Calc(const cv::Mat& src, cv::Mat& dst) {
    this->ConvertTo<Td>(
        this->TransformSqDist<Ts>(src,
                                  [](const Ts& val) -> bool {
                                    return std::abs(val) > std::numeric_limits<Ts>::epsilon();
                                  }),
        dst, [](const T& val) -> Td { return cv::saturate_cast<Td>(std::sqrt(val)); });
  }

  // 距離変換(距離の二乗))
  template <typename Ts, typename Td,
            std::enable_if_t<std::is_same<T, Td>{}, std::nullptr_t> = nullptr>
  void CalcSq(const cv::Mat& src, cv::Mat& dst) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src,
                                                  [](const Ts& val) -> bool {
                                                    return std::abs(val) >
                                                           std::numeric_limits<Ts>::epsilon();
                                                  }),
                        dst, [](const T& val) -> Td { return val; });
  }

  // 距離変換(距離の二乗))
  template <typename Ts, typename Td,
            std::enable_if_t<!std::is_same<T, Td>{}, std::nullptr_t> = nullptr>
  void CalcSq(const cv::Mat& src, cv::Mat& dst) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src,
                                                  [](const Ts& val) -> bool {
                                                    return std::abs(val) >
                                                           std::numeric_limits<Ts>::epsilon();
                                                  }),
                        dst, [](const T& val) -> Td { return cv::saturate_cast<Td>(val); });
  }

  // 距離変換(0のみ)
  template <typename Ts, typename Td,
            std::enable_if_t<std::is_same<T, Td>{}, std::nullptr_t> = nullptr>
  void CalcInv(const cv::Mat& src, cv::Mat& dst) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src,
                                                  [](const Ts& val) -> bool {
                                                    return std::abs(val) <=
                                                           std::numeric_limits<Ts>::epsilon();
                                                  }),
                        dst, [](const T& val) -> Td { return std::sqrt(val); });
  }

  // 距離変換(0のみ)
  template <typename Ts, typename Td,
            std::enable_if_t<!std::is_same<T, Td>{}, std::nullptr_t> = nullptr>
  void CalcInv(const cv::Mat& src, cv::Mat& dst) {
    this->ConvertTo<Td>(
        this->TransformSqDist<Ts>(src,
                                  [](const Ts& val) -> bool {
                                    return std::abs(val) <= std::numeric_limits<Ts>::epsilon();
                                  }),
        dst, [](const T& val) -> Td { return cv::saturate_cast<Td>(std::sqrt(val)); });
  }

  // 距離変換(0のみ、距離二乗)
  template <typename Ts, typename Td,
            std::enable_if_t<std::is_same<T, Td>{}, std::nullptr_t> = nullptr>
  void CalcInvSq(const cv::Mat& src, cv::Mat& dst) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src,
                                                  [](const Ts& val) -> bool {
                                                    return std::abs(val) <=
                                                           std::numeric_limits<Ts>::epsilon();
                                                  }),
                        dst, [](const T& val) -> Td { return val; });
  }

  // 距離変換(0のみ、距離二乗)
  template <typename Ts, typename Td,
            std::enable_if_t<!std::is_same<T, Td>{}, std::nullptr_t> = nullptr>
  void CalcInvSq(const cv::Mat& src, cv::Mat& dst) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src,
                                                  [](const Ts& val) -> bool {
                                                    return std::abs(val) <=
                                                           std::numeric_limits<Ts>::epsilon();
                                                  }),
                        dst, [](const T& val) -> Td { return cv::saturate_cast<Td>(val); });
  }

  // 距離変換(0以外)
  template <typename Ts>
  void Calc(const cv::Mat& src, cv::Mat& dst) {
    switch (CV_MAT_DEPTH(dst.type())) {
      case CV_8U:
        this->Calc<Ts, uchar>(src, dst);
        break;
      case CV_8S:
        this->Calc<Ts, char>(src, dst);
        break;
      case CV_16U:
        this->Calc<Ts, ushort>(src, dst);
        break;
      case CV_16S:
        this->Calc<Ts, short>(src, dst);
        break;
      case CV_32S:
        this->Calc<Ts, int>(src, dst);
        break;
      case CV_32F:
        this->Calc<Ts, float>(src, dst);
        break;
      case CV_64F:
        this->Calc<Ts, double>(src, dst);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 距離変換(距離の二乗))
  template <typename Ts>
  void CalcSq(const cv::Mat& src, cv::Mat& dst) {
    switch (CV_MAT_DEPTH(dst.type())) {
      case CV_8U:
        this->CalcSq<Ts, uchar>(src, dst);
        break;
      case CV_8S:
        this->CalcSq<Ts, char>(src, dst);
        break;
      case CV_16U:
        this->CalcSq<Ts, ushort>(src, dst);
        break;
      case CV_16S:
        this->CalcSq<Ts, short>(src, dst);
        break;
      case CV_32S:
        this->CalcSq<Ts, int>(src, dst);
        break;
      case CV_32F:
        this->CalcSq<Ts, float>(src, dst);
        break;
      case CV_64F:
        this->CalcSq<Ts, double>(src, dst);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 距離変換(0のみ)
  template <typename Ts>
  void CalcInv(const cv::Mat& src, cv::Mat& dst) {
    switch (CV_MAT_DEPTH(dst.type())) {
      case CV_8U:
        this->CalcInv<Ts, uchar>(src, dst);
        break;
      case CV_8S:
        this->CalcInv<Ts, char>(src, dst);
        break;
      case CV_16U:
        this->CalcInv<Ts, ushort>(src, dst);
        break;
      case CV_16S:
        this->CalcInv<Ts, short>(src, dst);
        break;
      case CV_32S:
        this->CalcInv<Ts, int>(src, dst);
        break;
      case CV_32F:
        this->CalcInv<Ts, float>(src, dst);
        break;
      case CV_64F:
        this->CalcInv<Ts, double>(src, dst);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 距離変換(0のみ、距離二乗)
  template <typename Ts>
  void CalcInvSq(const cv::Mat& src, cv::Mat& dst) {
    switch (CV_MAT_DEPTH(dst.type())) {
      case CV_8U:
        this->CalcInvSq<Ts, uchar>(src, dst);
        break;
      case CV_8S:
        this->CalcInvSq<Ts, char>(src, dst);
        break;
      case CV_16U:
        this->CalcInvSq<Ts, ushort>(src, dst);
        break;
      case CV_16S:
        this->CalcInvSq<Ts, short>(src, dst);
        break;
      case CV_32S:
        this->CalcInvSq<Ts, int>(src, dst);
        break;
      case CV_32F:
        this->CalcInvSq<Ts, float>(src, dst);
        break;
      case CV_64F:
        this->CalcInvSq<Ts, double>(src, dst);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 距離変換(0以外)
  void Calc(const cv::Mat& src, cv::Mat& dst) {
    switch (CV_MAT_DEPTH(src.type())) {
      case CV_8U:
        this->Calc<uchar>(src, dst);
        break;
      case CV_8S:
        this->Calc<char>(src, dst);
        break;
      case CV_16U:
        this->Calc<ushort>(src, dst);
        break;
      case CV_16S:
        this->Calc<short>(src, dst);
        break;
      case CV_32S:
        this->Calc<int>(src, dst);
        break;
      case CV_32F:
        this->Calc<float>(src, dst);
        break;
      case CV_64F:
        this->Calc<double>(src, dst);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 距離変換(距離の二乗))
  void CalcSq(const cv::Mat& src, cv::Mat& dst) {
    switch (CV_MAT_DEPTH(src.type())) {
      case CV_8U:
        this->CalcSq<uchar>(src, dst);
        break;
      case CV_8S:
        this->CalcSq<char>(src, dst);
        break;
      case CV_16U:
        this->CalcSq<ushort>(src, dst);
        break;
      case CV_16S:
        this->CalcSq<short>(src, dst);
        break;
      case CV_32S:
        this->CalcSq<int>(src, dst);
        break;
      case CV_32F:
        this->CalcSq<float>(src, dst);
        break;
      case CV_64F:
        this->CalcSq<double>(src, dst);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 距離変換(0のみ)
  void CalcInv(const cv::Mat& src, cv::Mat& dst) {
    switch (CV_MAT_DEPTH(src.type())) {
      case CV_8U:
        this->CalcInv<uchar>(src, dst);
        break;
      case CV_8S:
        this->CalcInv<char>(src, dst);
        break;
      case CV_16U:
        this->CalcInv<ushort>(src, dst);
        break;
      case CV_16S:
        this->CalcInv<short>(src, dst);
        break;
      case CV_32S:
        this->CalcInv<int>(src, dst);
        break;
      case CV_32F:
        this->CalcInv<float>(src, dst);
        break;
      case CV_64F:
        this->CalcInv<double>(src, dst);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 距離変換(0のみ、距離二乗)
  void CalcInvSq(const cv::Mat& src, cv::Mat& dst) {
    switch (CV_MAT_DEPTH(src.type())) {
      case CV_8U:
        this->CalcInvSq<uchar>(src, dst);
        break;
      case CV_8S:
        this->CalcInvSq<char>(src, dst);
        break;
      case CV_16U:
        this->CalcInvSq<ushort>(src, dst);
        break;
      case CV_16S:
        this->CalcInvSq<short>(src, dst);
        break;
      case CV_32S:
        this->CalcInvSq<int>(src, dst);
        break;
      case CV_32F:
        this->CalcInvSq<float>(src, dst);
        break;
      case CV_64F:
        this->CalcInvSq<double>(src, dst);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  //
  // 座標変換
  //

  // 最も近い座標を計算
  const cv::Point CalcPoint(const int x, const int y) {
    const int& _y = m_idxY[x * m_size.height + y];
    return {m_idxX[_y * m_size.width + x], _y};
  }

  // 最も近い座標を計算
  const cv::Point CalcPoint(const cv::Point& pos) { return CalcPoint(pos.x, pos.y); }

  // 最も近い座標を計算
  void CalcPoint(cv::Mat& points) {
    Assert(points.size() == m_size, "Poins-Data Size Error");

    tbb::tbb_for(points.size(), [&](const tbb::blocked_range2d<int>& range) {
      for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
        int x = std::begin(range.cols());
        cv::Point* ptrPoint = points.ptr<cv::Point>(y, x);
        const int* ptrY = &m_idxY[x * m_size.height + y];
        const int* ptrX = &m_idxX[x];
        for (; x < std::end(range.cols()); ++x, ++ptrPoint, ptrY += m_size.height, ++ptrX) {
          const int& _y = *ptrY;
          ptrPoint->x = ptrX[_y * m_size.width];
          ptrPoint->y = _y;
        }
      }
    });
  }

  const cv::Mat GetIndexX() const {
    return {m_size, cv::Type<std::int32_t>(), const_cast<std::int32_t*>(&m_idxX[0])};
  }
  const cv::Mat GetIndexY() const {
    return {m_size.width, m_size.height, cv::Type<std::int32_t>(),
            const_cast<std::int32_t*>(&m_idxY[0])};
  }
};

}}  // namespace imgproc::old

#ifdef _MSC_VER

// インポート・エクスポートマクロ
#ifndef _EXPORT_IMGPROC_
#if !defined(_MSC_VER) || defined(_LIB)
#define _EXPORT_IMGPROC_
#else
MSVC_WARNING_DISABLE(251)
#ifdef ImgProc_EXPORTS
#define _EXPORT_IMGPROC_ __declspec(dllexport)
#else
#define _EXPORT_IMGPROC_ __declspec(dllimport)
#endif
#endif
#endif

#ifndef _EXTERN_IMGPROC_
#ifdef ImgProc_EXPORTS
#define _EXTERN_IMGPROC_
#else
#define _EXTERN_IMGPROC_ extern
#endif
#endif

/// <summary>
/// 画像処理
/// </summary>
namespace imgproc {

//! インスタンス化抑制
_EXTERN_IMGPROC_ template class _EXPORT_IMGPROC_
    DistanceTransform_<float, tbb::auto_partitioner, tbb::cache_aligned_allocator>;
_EXTERN_IMGPROC_ template class _EXPORT_IMGPROC_
    DistanceTransform_<float, tbb::affinity_partitioner, tbb::cache_aligned_allocator>;
_EXTERN_IMGPROC_ template class _EXPORT_IMGPROC_
    DistanceTransformWithIndex_<float, tbb::auto_partitioner, tbb::cache_aligned_allocator>;
_EXTERN_IMGPROC_ template class _EXPORT_IMGPROC_
    DistanceTransformWithIndex_<float, tbb::affinity_partitioner, tbb::cache_aligned_allocator>;

}  // namespace imgproc

#endif
