#pragma once

#include <Common/CommonDef.h>
#include <Common/CommonFunction.h>
#include <Common/EigenConfig.h>
#include <Common/OpenCvConfig.h>
#include <Common/TbbConfig.h>

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <limits>
#include <vector>

#include <boost/format.hpp>
#include <boost/operators.hpp>

#include <Eigen/Eigen>

// 警告抑制解除
MSVC_WARNING_POP

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

namespace imgproc {
// ヒストグラム(演算子は連鎖継承)
template <typename T = uchar, int C = 3, int S = 3>
class Histogram : private boost::addable<
                      Histogram<T, C, S>,
                      boost::subtractable<
                          Histogram<T, C, S>,
                          boost::dividable2<Histogram<T, C, S>, double,
                                            boost::multipliable2<Histogram<T, C, S>, double>>>> {
 public:
  //! チャンネル辺りの量子化ビット数
  static constexpr int HistBit = S;
  //! ヒストグラム作成時のシフト量
  static constexpr int HistShift = sizeof(T) * 8 - HistBit;
  //! ヒストグラムの長さ
  static constexpr int HistLength = 1 << (HistBit * C);
  //! チャンネルマスク
  static constexpr int HistMask = (1 << (HistBit + 1)) - 1;

  //! インデックス計算
  static const int GetIdx(const cv::Vec<T, C>& val) {
    int ret = 0;
    int i = C - 1;
    ret += (val[i] >> HistShift) & HistMask;
    for (--i; i >= 0; --i) {
      ret <<= HistBit;
      ret += (val[i] >> HistShift) & HistMask;
    }
    return ret;
  }

  //! インデックス計算(逆)
  static const cv::Vec<T, C> GetIdxInv(int idx) {
    cv::Vec<T, C> ret;
    int i = 0;
    ret[i] = cv::saturate_cast<T>((idx & HistMask) << HistShift);
    for (++i; i < C; ++i) {
      idx >>= HistBit;
      ret[i] = cv::saturate_cast<T>((idx & HistMask) << HistShift);
    }
    return ret;
  }

  //! インデックス計算(逆)
  static const cv::Vec<T, 3> GetIdxInv3(int idx) {
    cv::Vec<T, 3> ret = {0, 0, 0};
    int i = 0;
    ret[i] = cv::saturate_cast<T>((idx & HistMask) << HistShift);
    for (++i; i < std::min(3, C); ++i) {
      idx >>= HistBit;
      ret[i] = cv::saturate_cast<T>((idx & HistMask) << HistShift);
    }
    return ret;
  }

  //! 入力データの型
  using InputType = T;
  //! 自身のタイプ設定
  using Type = Histogram<T, C, S>;
  //! ヒストグラムデータタイプ
  using DataType = Eigen::Matrix<float, HistLength, 1>;
  //! Vecタイプ
  using VecType = cv::Vec<T, C>;

 private:
  //! ヒストグラム
  DataType m_histogram;

  //! @name シリアライズ用設定
  //@{
  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned) {
    ar& boost::serialization::make_nvp("Histogram", m_histogram);
  }
  //@}

 public:
  // コンストラクタ
  Histogram()
      : m_histogram(DataType::Zero())
  //, m_mutex()
  {}

  // コンストラクタ
  Histogram(const cv::Mat& img, const double weight = 1.0)
      : m_histogram(DataType::Zero())
  //, m_mutex()
  {
    Add(img, weight);
  }

  // コピーコンストラクタ
  Histogram(const Histogram& rhs)
      : m_histogram(rhs.m_histogram)
  //, m_mutex()
  {}

  // コピーコンストラクタ
  template <typename T_>
  Histogram(const T_ (&rhs)[HistLength])
      : m_histogram()
  //, m_mutex()
  {
    for (int i = 0; i < HistLength; ++i) m_histogram[i] = static_cast<float>(rhs[i]);
  }

  // コピーコンストラクタ
  template <typename T_>
  Histogram(const std::array<T_, HistLength>& rhs)
      : m_histogram()
  //, m_mutex()
  {
    for (int i = 0; i < HistLength; ++i) m_histogram[i] = static_cast<float>(rhs[i]);
  }

  // コピーコンストラクタ
  template <typename T_>
  Histogram(const std::vector<T_>& rhs)
      : m_histogram()
  //, m_mutex()
  {
    int i = 0;
    for (; i < std::min(HistLength, static_cast<int>(rhs.size())); ++i)
      m_histogram[i] = static_cast<float>(rhs[i]);
    for (; i < HistLength; ++i) m_histogram[i] = 0.0f;
  }

  // スワップ
  void swap(Histogram& rhs) { std::swap(m_histogram, rhs.m_histogram); }

  // 代入演算子
  Histogram& operator=(const Histogram& rhs) {
    auto tmp = rhs;
    this->swap(tmp);
    return *this;
  }

  // 代入演算子
  template <typename T_>
  Histogram& operator=(const T_ (&rhs)[HistLength]) {
    Histogram tmp = rhs;
    this->swap(tmp);
    return *this;
  }

  // 代入演算子
  template <typename T_>
  Histogram& operator=(const std::array<T_, HistLength>& rhs) {
    Histogram tmp = rhs;
    this->swap(tmp);
    return *this;
  }

  // 代入演算子
  template <typename T_>
  Histogram& operator=(const std::vector<T_>& rhs) {
    Histogram tmp = rhs;
    this->swap(tmp);
    return *this;
  }

  // std::arrayへのキャスト
  operator std::array<double, HistLength>() {
    std::array<double, HistLength> ret;
    for (int i = 0; i < HistLength; ++i) ret[i] = m_histogram[i];
    return ret;
  }

  // 加算代入
  Histogram& operator+=(const Histogram& rhs) {
    m_histogram += rhs.m_histogram;
    return *this;
  }

  // 減算代入
  Histogram& operator-=(const Histogram& rhs) {
    m_histogram = (m_histogram - rhs.m_histogram).cwiseMax(0);
    return *this;
  }

  // 商算代入
  Histogram& operator/=(const double n) {
    m_histogram /= static_cast<float>(n);
    return *this;
  }

  // 乗算代入
  Histogram& operator*=(const double n) {
    m_histogram *= static_cast<float>(n);
    return *this;
  }

  // []演算子
  const double operator[](const int idx) const {
    Assert(0 <= idx && idx < HistLength, (boost::format("index = %d") % idx).str());
    return m_histogram[idx];
  }

  // 削除
  void Clear() { m_histogram.setZero(); }

  // 追加
  void Add(int idx, const double weight = 1.0) { m_histogram[idx] += static_cast<float>(weight); }

  // 追加
  void Add(const typename Histogram::VecType& val, const double weight = 1.0) {
    Add(GetIdx(val), weight);
  }

  // 追加
  void Add(const cv::Mat& img, const double weight = 1.0) {
    Assert((img.channels() == C) && (img.elemSize1() == sizeof(T)));

    // 全画素のヒストグラム作成
    m_histogram =
        tbb::tbb_reduce(img.size(), Histogram::DataType(Histogram::DataType::Zero()),
                        [&](const tbb::blocked_range2d<int>& range,
                            Histogram::DataType hist) -> Histogram::DataType {
                          for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
                            int x = std::begin(range.cols());
                            const VecType* ptr = img.ptr<VecType>(y, x);
                            for (; x < std::end(range.cols()); ++x, ++ptr)
                              hist[GetIdx(*ptr)] += static_cast<float>(weight);
                          }
                          return hist;
                        },
                        [](const Histogram::DataType& a,
                           const Histogram::DataType& b) -> Histogram::DataType { return a + b; });
  }

  // 正規化
  void Normalize() { m_histogram /= m_histogram.sum(); }

  // バタチャリア係数による比較
  static const double CompareBhattacharyyaCoeff(const Histogram& lhs, const Histogram& rhs) {
    DataType a = (lhs.m_histogram.array() + std::numeric_limits<float>::epsilon());
    a /= a.sum();
    DataType b = (rhs.m_histogram.array() + std::numeric_limits<float>::epsilon());
    b /= b.sum();
    return a.cwiseProduct(b).cwiseSqrt().sum();
  }
  const double CompareBhattacharyyaCoeff(const Histogram& hist) const {
    return Histogram::CompareBhattacharyyaCoeff(*this, hist);
  }

  // 最小値選択による比較
  static const double CompareIntersection(const Histogram& lhs, const Histogram& rhs) {
    const DataType a = lhs.m_histogram / lhs.m_histogram.sum();
    const DataType b = rhs.m_histogram / rhs.m_histogram.sum();
    return a.cwiseMin(b).sum();
  }
  const double CompareIntersection(const Histogram& hist) const {
    return Histogram::CompareIntersection(*this, hist);
  }

  // カイ2乗による比較
  static const double CompareChiSquare(const Histogram& lhs, const Histogram& rhs) {
    const DataType a = lhs.m_histogram / lhs.m_histogram.sum();
    const DataType b = rhs.m_histogram / rhs.m_histogram.sum();
    return (a - b).array().pow(2.0).sum();
  }
  const double CompareChiSquare(const Histogram& hist) const {
    return Histogram::CompareChiSquare(*this, hist);
  }

  // ヒストグラムインデックス作成
  static void CreateIndexImage(const cv::Mat& src, cv::Mat& dst) {
    Assert((src.channels() == C) && (src.elemSize1() == sizeof(T)) && (src.size() == dst.size()) &&
           dst.type() == CV_32SC1);

    tbb::tbb_for(src.size(), [&](const tbb::blocked_range2d<int>& range) {
      for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
        int x = std::begin(range.cols());
        const VecType* ptrSrc = src.ptr<VecType>(y, x);
        int* ptrDst = dst.ptr<int>(y, x);
        for (; x < std::end(range.cols()); ++x, ++ptrSrc, ++ptrDst) *ptrDst = GetIdx(*ptrSrc);
      }
    });
  }

  // デバッグ画像取得
  void CreateDebugImage(cv::Mat& dbg,
                        const std::function<const cv::Vec3b&(const cv::Vec3b&)>& func) const {
    static const cv::Size dbgSize = Histogram::GetDebugSize();

    if (dbg.size() != dbgSize || dbg.type() != CV_8UC3) dbg.create(dbgSize, CV_8UC3);

    // 最大値取得
    const float maxVal = m_histogram.maxCoeff();

    // 画像初期化
    dbg = 0;
    if (maxVal > std::numeric_limits<float>::epsilon()) {
      constexpr bool flagCh = (C == 3);
      tbb::tbb_for(0, dbgSize.width, [&](const int x) {
        int histVal = commonutility::RangeModify(cvFloor(m_histogram[x] / maxVal * dbgSize.height),
                                                 0, dbgSize.height);
        for (int y = dbgSize.height - histVal; y < dbgSize.height; ++y) MSVC_WARNING_PUSH
        MSVC_WARNING_DISABLE(127)
        dbg.at<cv::Vec3b>(y, x) = flagCh ? func(GetIdxInv3(x)) : cv::Vec3b(255, 255, 255);
        MSVC_WARNING_POP
      });
    }
  }
  void CreateDebugImage(cv::Mat& dbg) const {
    this->CreateDebugImage(dbg, [](const cv::Vec3b& pix) -> const cv::Vec3b& { return pix; });
  }

  // マルチデバッグ画像取得
  template <int N>
  static void CreateMultiDebugImage(const std::array<Histogram, N>& histograms, cv::Mat& dbg,
                                    const std::function<const cv::Vec3b&(const cv::Vec3b&)>& func,
                                    const cv::Scalar& color = cv::Scalar::all(255)) {
    static const cv::Size dbgSize = Histogram::GetMultiDebugSize(N);
    static const cv::Size histSize = Histogram::GetDebugSize();

    if (dbg.size() != dbgSize || dbg.type() != CV_8UC3) dbg.create(dbgSize, CV_8UC3);

    std::array<cv::Mat, N> roiDbgs;
    int i = 0;
    rectangle(dbg, cv::Rect(cv::Point(0, 0), dbgSize), color);
    roiDbgs[i] = cv::Mat(dbg, cv::Rect(cv::Point(1, 1), histSize));
    for (++i; i < N; ++i) {
      const int yPos = (histSize.height + 1) * i;
      cv::line(dbg, cv::Point(0, yPos), cv::Point(histSize.width + 1, yPos), color);
      roiDbgs[i] = cv::Mat(dbg, cv::Rect(cv::Point(1, (histSize.height + 1) * i + 1), histSize));
    }
    tbb::tbb_for(0, N, [&](const int i) { histograms[i].CreateDebugImage(roiDbgs[i], func); });
  }
  template <int N>
  static void CreateMultiDebugImage(const std::array<Histogram, N>& histograms, cv::Mat& dbg,
                                    const cv::Scalar& color = cv::Scalar::all(255)) {
    Histogram::CreateMultiDebugImage<N>(
        histograms, dbg, [](const cv::Vec3b& pix) -> const cv::Vec3b& { return pix; }, color);
  }

  // デバッグ画像サイズ
  static const cv::Size GetDebugSize() { return {HistLength, cvFloor(HistLength / 3.0)}; }

  // マルチデバッグ画像サイズ
  static const cv::Size GetMultiDebugSize(const int n) {
    const cv::Size histSize = Histogram::GetDebugSize();
    return {histSize.width + 2, (histSize.height + 1) * n + 1};
  }
};

using ColorHist = Histogram<>;
//using GrayHist = Histogram<uchar, 1, 8>;
//using MultiHist = std::array<GrayHist, 3>;

}  // namespace imgproc

#ifdef _MSC_VER

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
_EXTERN_IMGPROC_ template class _EXPORT_IMGPROC_ Histogram<uchar, 3, 3>;

}  // namespace imgproc

#endif
