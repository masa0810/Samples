#pragma once

#include "ImageCollection.h"
#include "OpenCvConfig.h"

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <cstddef>
#include <cmath>

#include <algorithm>
#include <functional>
#include <limits>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <vector>

#include <boost/container_hash/hash.hpp>
#include <boost/operators.hpp>

#include <opencv2/core.hpp>

#include <tbb/tbb.h>

// 警告抑制解除
MSVC_WARNING_POP

/// <summary>
/// 画像処理
/// </summary>
namespace imgproc {

/// <summary>
/// 詳細実装
/// </summary>
namespace detail {

/// <summary>
/// コピーしないメンバー定義のためのクラス
/// </summary>
/// <remarks>
/// <para>std::mutex などコピーやムーブが禁止されているクラスをメンバーとして持った場合、</para>
/// <para>コピーコンストラクタなどのコンパイラ定義のメソッドを自分で定義しなくてはならない。</para>
/// <para>その定義を簡略化するためのクラス。</para>
/// </remarks>
/// @tparam Args メンバータイプ
template <typename... Args>
struct NonCopyMembers_ {
  //! メンバ
  std::tuple<Args...> NcMembers = {};

#pragma region デフォルトメソッド定義

  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  NonCopyMembers_() = default;

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  NonCopyMembers_(const NonCopyMembers_&) {}

  /// <summary>
  /// 代入演算子
  /// </summary>
  NonCopyMembers_& operator=(const NonCopyMembers_&) { return *this; }

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  NonCopyMembers_(NonCopyMembers_&&) {}

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  NonCopyMembers_& operator=(NonCopyMembers_&&) { return *this; }

  /// <summary>
  /// デストラクタ
  /// </summary>
  virtual ~NonCopyMembers_() = default;

#pragma endregion
};

template <typename T, template <typename...> class Alloc_>
struct Table_ {
  template <typename U>
  using VecType = std::vector<U, Alloc_<U>>;
  int Size = 0;
  VecType<T> Sqr = {};
  VecType<int> Sat = {};
  VecType<T> Inv = {};
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
};

template <typename T, template <typename...> class Alloc_>
std::tuple<std::vector<int, Alloc_<int>>&, std::vector<T, Alloc_<T>>&> GetTlsBuffer(
    const int bufSize) {
  using IdxBufType = std::vector<int, Alloc_<int>>;
  using TmpBufType = std::vector<T, Alloc_<T>>;
  thread_local auto _bufSize = 0;
  thread_local IdxBufType bufIdx;
  thread_local TmpBufType bufTmp;
  if (_bufSize < bufSize) {
    _bufSize = bufSize;
    bufIdx.resize(_bufSize);
    bufTmp.resize(_bufSize << 1);
  }
  return std::make_tuple(std::ref(bufIdx), std::ref(bufTmp));
}

/// <summary>
/// 1次元の距離変換
/// </summary>
/// <param name="size">[in] サイズ</param>
/// <param name="byteStep">[in] ステップ(byte)</param>
/// <param name="f">[in] 入力データのポインタ</param>
/// <param name="ptrD0">[out] 出力データのポインタ</param>
/// <param name="ptrI0">[out] 最短インデックスのポインタ</param>
/// <param name="v">[out] インデックスバッファのポインタ</param>
/// <param name="ptrZ0">[out] 一時バッファのポインタ</param>
/// @tparam T 入出力・一時バッファの型
template <typename T, typename Ts, typename F>
void TransformSimple1D(const T* const sqrTbl, const int* const satTbl, const int size,
                       const int byteStep, const Ts* const s, T* ptrD0, int* ptrI0, int* const v,
                       const F& binarization) {
  const auto sizeM1 = size - 1;
  {
    const auto* ptrS = s + sizeM1;
    auto* ptrV = v + sizeM1;
    auto dist = sizeM1;
    for (auto i = sizeM1; i >= 0; --i, --ptrS, --ptrV) {
      *ptrV = dist = binarization(*ptrS) ? (dist + 1) : 0;
    }
  }
  {
    auto* ptrV = v;
    auto dist = sizeM1;
    for (auto i = 0; i < size; ++i, ++ptrV, ptrD0 = cv::PtrByteInc(ptrD0, byteStep), ++ptrI0) {
      *ptrV = dist = dist + 1 - satTbl[dist - *ptrV];
      //*ptrD0 = sqrTbl[dist];
      *ptrD0 = dist * dist;
      *ptrI0 = *ptrV;
    }
  }
}

/// <summary>
/// 1次元の距離変換(並列実行対応)
/// </summary>
/// <param name="bufSize">[in] バッファサイズ</param>
/// <param name="size">[in] サイズ</param>
/// <param name="byteStep">[in] ステップ(byte)</param>
/// <param name="f">[in] 入力データのポインタ</param>
/// <param name="ptrD0">[out] 出力データのポインタ</param>
/// <param name="ptrI0">[out] 最短インデックスのポインタ</param>
/// <remarks>距離変換処理を並列実行するために、スレッド毎のバッファを用意する。</remarks>
/// @tparam T 入出力・一時バッファの型
/// @tparam Alloc_ バッファのアロケータ
template <typename T, template <typename...> class Alloc_, typename Ts, typename F>
void TransformSimple1D(const Table_<T, Alloc_>& tbl, const int size, const int byteStep,
                       const Ts* const s, T* ptrD0, int* ptrI0, const F& binarization) {
  auto buf = GetTlsBuffer<T, Alloc_>(tbl.Size);
  auto& bufIdx = std::get<0>(buf);
  TransformSimple1D<T, Ts, F>(&tbl.Sqr[0], &tbl.Sat[tbl.SatStart], size, byteStep, s, ptrD0, ptrI0,
                              &bufIdx[0], binarization);
}

/// <summary>
/// 1次元の距離変換
/// </summary>
/// <param name="size">[in] サイズ</param>
/// <param name="byteStep">[in] ステップ(byte)</param>
/// <param name="f">[in] 入力データのポインタ</param>
/// <param name="ptrD0">[out] 出力データのポインタ</param>
/// <param name="ptrI0">[out] 最短インデックスのポインタ</param>
/// <param name="v">[out] インデックスバッファのポインタ</param>
/// <param name="ptrZ0">[out] 一時バッファのポインタ</param>
/// @tparam T 入出力・一時バッファの型
template <typename T, typename Td, typename F>
void Transform1D(const T* const sqrTbl, const T* const invTbl, const int size, const int byteStep,
                 const T* const f, Td* ptrD0, int* ptrI0, int* const v, T* ptrZ0, const F& conv) {
  const auto* ptrF1 = f + 1;
  auto* ptrV0 = v;
  auto* ptrZ1 = ptrZ0 + 1;

  *ptrV0 = 0;
  *ptrZ0 = std::numeric_limits<T>::lowest();
  *ptrZ1 = std::numeric_limits<T>::max();
  for (auto q = 1; q < size; ++q, ++ptrF1) {
    const auto A = (*ptrF1) + (q * q);
    const auto& v1 = *ptrV0;
    auto s = (A - (f[v1] + sqrTbl[v1])) * invTbl[q - v1];
    while (s <= *ptrZ0) {
      --ptrV0;
      --ptrZ0;

      const auto& v2 = *ptrV0;
      s = (A - (f[v2] + sqrTbl[v2])) * invTbl[q - v2];
    }
    ++ptrV0;
    ++ptrZ0;

    *ptrV0 = q;
    *ptrZ0 = s;
    *(ptrZ0 + 1) = std::numeric_limits<T>::max();
  }

  ptrV0 = v;
  for (auto q = 0; q < size; ++q, ptrD0 = cv::PtrByteInc(ptrD0, byteStep), ++ptrI0) {
    while (*ptrZ1 < q) {
      ++ptrV0;
      ++ptrZ1;
    }
    const auto& refV = *ptrV0;
    *ptrD0 = cv::saturate_cast<Td>(conv(sqrTbl[std::abs(q - refV)] + f[refV]));
    *ptrI0 = refV;
  }
}

/// <summary>
/// 1次元の距離変換(並列実行対応)
/// </summary>
/// <param name="bufSize">[in] バッファサイズ</param>
/// <param name="size">[in] サイズ</param>
/// <param name="byteStep">[in] ステップ(byte)</param>
/// <param name="f">[in] 入力データのポインタ</param>
/// <param name="ptrD0">[out] 出力データのポインタ</param>
/// <param name="ptrI0">[out] 最短インデックスのポインタ</param>
/// <remarks>距離変換処理を並列実行するために、スレッド毎のバッファを用意する。</remarks>
/// @tparam T 入出力・一時バッファの型
/// @tparam Alloc_ バッファのアロケータ
template <typename T, template <typename...> class Alloc_, typename Td, typename F>
void Transform1D(const Table_<T, Alloc_>& tbl, const int size, const int byteStep, const T* const f,
                 T* ptrD0, int* ptrI0, const F& conv) {
  auto buf = GetTlsBuffer<T, Alloc_>(tbl.Size);
  auto& bufIdx = std::get<0>(buf);
  auto& bufTmp = std::get<1>(buf);
  Transform1D<T>(&tbl.Sqr[0], &tbl.Inv[0], size, byteStep, f, ptrD0, ptrI0, &bufIdx[0],
                 &bufTmp[size], conv);
}

/// <summary>
/// 2値化&2次元の距離変換
/// </summary>
/// <param name="bufSize">[in] バッファサイズ</param>
/// <param name="src">[in] 入力</param>
/// <param name="sqDist">[out] 出力(二乗距離)</param>
/// <param name="idxX">[out] インデックスバッファ(x)</param>
/// <param name="idxY">[out] インデックスバッファ(y)</param>
/// <param name="bufSqDist">[out] 二乗距離計算用バッファ</param>
/// <param name="paX">[in] パーティショナ(x方向計算用)</param>
/// <param name="paY">[in] パーティショナ(y方向計算用)</param>
/// <param name="binarization">[in] 入力画像を2値化する関数</param>
/// <remarks>
/// <para>距離変換アルゴリズムの特性上、1行または1列毎に距離計算を行う。</para>
/// <para>この実装では、先に列方向(x方向)で距離計算を行ってから行方向(y方向)の距離計算を行う。</para>
/// <para>高速化のために方向毎に処理を並列化するが、最初の並列化(x方向)の際に</para>
/// <para>画像の2値化処理も同時に行う。2値化の方法は関数オブジェクトで指定する。</para>
/// <para>また、距離計算の際に算出される最短インデックスを保持しておくことにより、</para>
/// <para>後から最短座標を計算できるようになっている。</para>
/// </remarks>
/// @tparam T 出力・一時バッファの型
/// @tparam Alloc_ バッファのアロケータ
/// @tparam P パーティショナの型
/// @tparam Ts 入力画像の型
/// @tparam F 2値化関数オブジェクトの型
template <typename T, template <typename...> class Alloc_, typename P, typename Ts, typename Td,
          typename Fb, typename Fc>
void Transform2D(const Table_<T, Alloc_>& tbl, const cv::Mat& src, cv::Mat& dst, cv::Mat idxX,
                 cv::Mat idxY, cv::Mat bufSqDist, P& paX, P& paY, const Fb& binarization,
                 const Fc& conv) {
  // 行単位で並列化
  tbb::parallel_for(0, src.rows,
                    [tbl, size = src.cols, byteStep = bufSqDist.step[0], &src, &idxX, &bufSqDist,
                     &binarization](const auto y) {
                      // 距離変換
                      const auto* const ptrSrc = src.ptr<Ts>(y);
                      auto* ptrBufSqDist = bufSqDist.ptr<T>(0, y);
                      auto* ptrIdxX = idxX.ptr<int>(y);
                      TransformSimple1D<T, Alloc_, Ts, Fb>(tbl, size, byteStep, ptrSrc,
                                                           ptrBufSqDist, ptrIdxX, binarization);
                    },
                    paX);

  // 列単位で並列化
  tbb::parallel_for(
      0, dst.cols,
      [tbl, size = dst.rows, byteStep = dst.step[0], &dst, &idxY, &bufSqDist, &conv](const auto x) {
        const auto* const ptrBufSqDist = bufSqDist.ptr<T>(x);
        auto* ptrDst = dst.ptr<T>(0, x);
        auto* ptrIdxY = idxY.ptr<int>(x);
        Transform1D<T, Alloc_, Td, Fc>(tbl, size, byteStep, ptrBufSqDist, ptrDst, ptrIdxY, conv);
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
/// @tparam Alloc_ アロケータの型
template <typename Tf = float, typename P = tbb::affinity_partitioner,
          template <typename...> class Alloc_ = tbb::cache_aligned_allocator>
class DistanceTransform_ : public detail::NonCopyMembers_<P, P>,
                           boost::equality_comparable<DistanceTransform_<Tf, P, Alloc_>,
                                                      DistanceTransform_<Tf, P, Alloc_>> {
  static_assert(std::is_floating_point<Tf>{}, "template parameter Tf must be floatng point type");

 public:
  //! 浮動小数点タイプ
  using FloatType = Tf;
  //! パーティショナタイプ
  using PartitionerType = P;

 private:
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
    auto hash = boost::hash_value(rhs.m_bufSize);
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
    ar& boost::serialization::make_nvp("BufSizeTLS", m_bufSize);
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
    detail::Transform2D<FloatType, Alloc_, PartitionerType, Ts, Td, Fb, Fc>(
        m_table, src, dst, m_imgBuf.Mat<Buf::X>(), m_imgBuf.Mat<Buf::Y>(), m_imgBuf.Mat<Buf::Buf>(),
        std::get<0>(NcMembers), std::get<1>(NcMembers), binarization, conv);
  }

  /// <summary>
  /// 距離変換
  /// </summary>
  /// <param name="src">[in] 入力画像</param>
  /// <param name="dst">[dst] 出力画像</param>
  /// <param name="binarization">[in] 入力画像を2値化する関数</param>
  /// <remarks>事前に Init を実行し、初期化を行っておかなければならない。</remarks>
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
  /// <param name="binarization">[in] 入力画像を2値化する関数</param>
  /// <remarks>事前に Init を実行し、初期化を行っておかなければならない。</remarks>
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
  /// <param name="binarization">[in] 入力画像を2値化する関数</param>
  /// <remarks>事前に Init を実行し、初期化を行っておかなければならない。</remarks>
  /// @tparam Ts 入力画像の型
  /// @tparam Td 出力画像の型
  template <typename Ts, typename Td>
  void CalcSq(const cv::Mat& src, cv::Mat& dst) {
    this->CalcSq<Ts, Td>(src, dst, [](const auto val) { return val != 0; });
  }
};

namespace old {
template <typename T = double, typename P = tbb::auto_partitioner>
class DistanceTransform_ {
 public:
  using PartitionerType = P;

 private:
  //! バッファは浮動小数点型のみ
  static_assert(std::is_floating_point<T>::value,
                "template parameter T must be floatng point type");

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
    tbb::parallel_for(0, size.height,
                      [&](const int y) {
                        const int idx = y * size.width;

                        // ※vs2010のラムダ式の不具合により、template引数も記述
                        DistanceTransform_<T, P>::Transform1D(size.width, size.height, &data[idx],
                                                              &work[y], &idxX[idx], &workV[idx],
                                                              &workZ[idx + y]);
                      },
                      paX);

    // 縦方向の距離変換
    tbb::parallel_for(0, size.width,
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
    // 2値データを浮動小数点型に変換
    cv::Mat buf(m_size, cv::Type<T>(), &m_sqDist[0]);
    tbb::parallel_for(tbb::blocked_range2d<int>(0, src.rows, 0, src.cols),
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
    // 出力データに変換
    tbb::parallel_for(tbb::blocked_range2d<int>(0, dst.rows, 0, dst.cols),
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

  // 距離変換
  template <typename Ts, typename Td, typename F1, typename F2>
  void CalcCompCast(const cv::Mat& src, cv::Mat& dst, const F1& comp, const F2& cast) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src, comp), dst, cast);
  }
};
}  // namespace old

}  // namespace imgproc
