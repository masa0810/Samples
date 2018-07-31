#pragma once

#include <Common/CommonDef.h>
#include <Common/OpenCvConfig.h>
#include <Common/TbbConfig.h>

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <cstdint>

#include <iterator>
#include <type_traits>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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

// 色変換テーブルタイプ
STREAMABLE_ENUM_CLASS(ConvertColorType,
                      // BGR->YUV
                      (BGRtoYUV)
                      // BGR->HSV
                      (BGRtoHSV)
                      // BGR->LAB
                      (BGRtoLAB)
                      // YUV->BGR
                      (YUVtoBGR)
                      // HSV->BGR
                      (HSVtoBGR)
                      // LAB->BGR
                      (LABtoBGR))

// BGR->YUV
template <imgproc::ConvertColorType ConvType,
          std::enable_if_t<ConvType == ConvertColorType::BGRtoYUV, std::nullptr_t> = nullptr>
void CvtColor(const cv::Mat& src, cv::Mat& dst) {
  cv::cvtColor(src, dst, cv::COLOR_BGR2YUV);
}
// BGR->HSV
template <ConvertColorType ConvType,
          std::enable_if_t<ConvType == ConvertColorType::BGRtoHSV, std::nullptr_t> = nullptr>
void CvtColor(const cv::Mat& src, cv::Mat& dst) {
  cv::cvtColor(src, dst, cv::COLOR_BGR2HSV_FULL);
}
// BGR->LAB
template <ConvertColorType ConvType,
          std::enable_if_t<ConvType == ConvertColorType::BGRtoLAB, std::nullptr_t> = nullptr>
void CvtColor(const cv::Mat& src, cv::Mat& dst) {
  cv::cvtColor(src, dst, cv::COLOR_BGR2Lab);
}
// YUV->BGR
template <ConvertColorType ConvType,
          std::enable_if_t<ConvType == ConvertColorType::YUVtoBGR, std::nullptr_t> = nullptr>
void CvtColor(const cv::Mat& src, cv::Mat& dst) {
  cv::cvtColor(src, dst, cv::COLOR_YUV2BGR);
}
// HSV->BGR
template <ConvertColorType ConvType,
          std::enable_if_t<ConvType == ConvertColorType::HSVtoBGR, std::nullptr_t> = nullptr>
void CvtColor(const cv::Mat& src, cv::Mat& dst) {
  cv::cvtColor(src, dst, cv::COLOR_HSV2BGR_FULL);
}
// LAB->BGR
template <ConvertColorType ConvType,
          std::enable_if_t<ConvType == ConvertColorType::LABtoBGR, std::nullptr_t> = nullptr>
void CvtColor(const cv::Mat& src, cv::Mat& dst) {
  cv::cvtColor(src, dst, cv::COLOR_Lab2BGR);
}

// 色変換テーブル
template <ConvertColorType ConvType>
class ConvertColor {
  using DefaultPartitionerType = const tbb::auto_partitioner;

  static constexpr size_t ValueSize = 256;
  static constexpr size_t ImageSize = ValueSize * ValueSize;
  static constexpr size_t TableSize = ImageSize * ValueSize;

  // デフォルトパーティショナー取得
  static DefaultPartitionerType& GetDefaultPartitioner() {
    static DefaultPartitionerType instance;
    return instance;
  }

  //! 色テーブル
  std::vector<cv::Vec3b> m_colorTable = std::vector<cv::Vec3b>(TableSize);

  // コンストラクタ
  ConvertColor() {
    // プレーン方向にループ
    for (auto i = 0; i < ValueSize; ++i) {
      cv::Mat base(ValueSize, ValueSize, cv::Type<std::uint8_t, 3>());

      // ベース画像の画素値初期化
      tbb::tbb_for(tbb::blocked_range2d<int>(0, ValueSize, 0, ValueSize), [i, &base](
                                                                              const auto& range) {
        const auto& rows = range.rows();
        const auto& cols = range.cols();
        auto y = std::begin(rows);
        for (const auto ye = std::end(rows); y < ye; ++y) {
          auto x = std::begin(cols);
          auto* ptr = base.ptr<cv::Vec3b>(y, x);
          for (const auto xe = std::end(cols); x < xe; ++x, ++ptr) {
            *ptr = cv::Vec3b(cv::saturate_cast<std::uint8_t>(x), cv::saturate_cast<std::uint8_t>(y),
                             cv::saturate_cast<std::uint8_t>(i));
          }
        }
      });

      //
      // 変換処理
      //
      const auto idx = i * ValueSize * ValueSize;
      cv::Mat buf(ValueSize, ValueSize, cv::Type<std::uint8_t, 3>(), &m_colorTable[idx]);
      CvtColor<ConvType>(base, buf);
    }
  }

  // テーブル取得(整数値)
  template <typename T, std::enable_if_t<std::is_integral<T>{}, std::nullptr_t> = nullptr>
  const cv::Vec3b& GetTbl(const T val1, const T val2, const T val3) const {
    return m_colorTable[val3 * ImageSize + val2 * ValueSize + val1];
  }

  // テーブル取得(浮動小数点型)
  template <typename T, std::enable_if_t<std::is_floating_point<T>{}, std::nullptr_t> = nullptr>
  const cv::Vec3b& GetTbl(const T val1, const T val2, const T val3) const {
    const auto val1_ = cv::saturate_cast<int>(val1);
    const auto val2_ = cv::saturate_cast<int>(val2);
    const auto val3_ = cv::saturate_cast<int>(val3);
    return this->GetTbl(val1_, val2_, val3_);
  }

 public:
#pragma region デフォルトメソッド定義

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  ConvertColor(const ConvertColor&) = delete;

  /// <summary>
  /// 代入演算子
  /// </summary>
  /// <returns>自分自身</returns>
  ConvertColor& operator=(const ConvertColor&) = delete;

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  ConvertColor(ConvertColor&&) = delete;

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  /// <returns>自分自身</returns>
  ConvertColor& operator=(ConvertColor&&) = delete;

  /// <summary>
  /// デストラクタ
  /// </summary>
  ~ConvertColor() = default;

#pragma endregion

  // 変換
  template <typename Td, typename Ts,
            std::enable_if_t<std::is_same<std::uint8_t, Td>{}, std::nullptr_t> = nullptr>
  const cv::Vec<Td, 3>& Conv(const Ts val1, const Ts val2, const Ts val3) const {
    return this->GetTbl<Ts>(val1, val2, val3);
  }

  // 変換
  template <typename Td, typename Ts,
            std::enable_if_t<std::is_same<std::uint8_t, Td>{}, std::nullptr_t> = nullptr>
  const cv::Vec<Td, 3>& Conv(const cv::Vec<Ts, 3>& val) const {
    return this->Conv<Td, Ts>(val[0], val[1], val[2]);
  }

  // 変換
  template <typename Td, typename Ts,
            std::enable_if_t<!std::is_same<std::uint8_t, Td>{}, std::nullptr_t> = nullptr>
  cv::Vec<Td, 3> Conv(const Ts val1, const Ts val2, const Ts val3) const {
    const auto& convVal = this->GetTbl<Ts>(val1, val2, val3);
    return {cv::saturate_cast<Td>(convVal[0]), cv::saturate_cast<Td>(convVal[1]),
            cv::saturate_cast<Td>(convVal[2])};
  }

  // 変換
  template <typename Td, typename Ts,
            std::enable_if_t<!std::is_same<std::uint8_t, Td>{}, std::nullptr_t> = nullptr>
  cv::Vec<Td, 3> Conv(const cv::Vec<Ts, 3>& val) const {
    return this->Conv<Td, Ts>(val[0], val[1], val[2]);
  }

  // 変換
  template <typename Ts, typename Td>
  void Conv(const cv::Vec<Ts, 3>& src, cv::Vec<Td, 3>& dst) const {
    dst = this->Conv<Td, Ts>(src);
  }

  // 変換
  const cv::Vec3b& operator()(const int val1, const int val2, const int val3) const {
    return this->Conv<std::uint8_t, int>(val1, val2, val3);
  }

  // 変換
  const cv::Vec3b& operator()(const cv::Vec3b& val) const {
    return this->Conv<std::uint8_t, std::uint8_t>(val);
  }

  // 変換
  template <typename Ts, typename Td>
  void operator()(const cv::Vec<Ts, 3>& src, cv::Vec<Td, 3>& dst) const {
    this->Conv<Ts, Td>(src, dst);
  }

  // 画像変換
  template <typename Ts, typename Td, bool EnableParallel, typename P>
  void Run(const cv::Mat& src, cv::Mat& dst, P& pa) const {
    tbb::tbb_for_if<EnableParallel>(src.size(),
                                    [&src, &dst, this](const auto& range) {
                                      const auto& rows = range.rows();
                                      const auto& cols = range.cols();
                                      auto y = std::begin(rows);
                                      for (const auto ye = std::end(rows); y < ye; ++y) {
                                        auto x = std::begin(cols);
                                        const auto* ptrSrc = src.ptr<cv::Vec<Ts, 3>>(y, x);
                                        auto* ptrDst = dst.ptr<cv::Vec<Td, 3>>(y, x);
                                        for (const auto xe = std::end(cols); x < xe;
                                             ++x, ++ptrSrc, ++ptrDst)
                                          this->Conv<Ts, Td>(*ptrSrc, *ptrDst);
                                      }
                                    },
                                    pa);
  }

  // 画像変換
  template <typename Ts, typename Td, bool EnableParallel>
  void Run(const cv::Mat& src, cv::Mat& dst) const {
    this->Run<Ts, Td, EnableParallel, DefaultPartitionerType>(
        src, dst, ConvertColor::GetDefaultPartitioner());
  }

  // 画像変換
  template <typename Ts, typename Td, typename P>
  void Run(const cv::Mat& src, cv::Mat& dst, P& pa) const {
    this->Run<Ts, Td, true, P>(src, dst, pa);
  }

  // 画像変換
  template <typename Ts, typename Td>
  void Run(const cv::Mat& src, cv::Mat& dst) const {
    this->Run<Ts, Td, true>(src, dst);
  }

  // 画像変換
  template <typename Ts>
  void Run(const cv::Mat& src, cv::Mat& dst) const {
    switch (CV_MAT_DEPTH(dst.type())) {
      case CV_8U:
        this->Run<Ts, std::uint8_t>(src, dst);
        break;
      case CV_8S:
        this->Run<Ts, std::int8_t>(src, dst);
        break;
      case CV_16U:
        this->Run<Ts, std::uint16_t>(src, dst);
        break;
      case CV_16S:
        this->Run<Ts, std::int16_t>(src, dst);
        break;
      case CV_32S:
        this->Run<Ts, std::int32_t>(src, dst);
        break;
      case CV_32F:
        this->Run<Ts, float>(src, dst);
        break;
      case CV_64F:
        this->Run<Ts, double>(src, dst);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 画像変換
  void Run(const cv::Mat& src, cv::Mat& dst) const {
    switch (CV_MAT_DEPTH(src.type())) {
      case CV_8U:
        this->Run<std::uint8_t>(src, dst);
        break;
      case CV_8S:
        this->Run<std::int8_t>(src, dst);
        break;
      case CV_16U:
        this->Run<std::uint16_t>(src, dst);
        break;
      case CV_16S:
        this->Run<std::int16_t>(src, dst);
        break;
      case CV_32S:
        this->Run<std::int32_t>(src, dst);
        break;
      case CV_32F:
        this->Run<float>(src, dst);
        break;
      case CV_64F:
        this->Run<double>(src, dst);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 画像変換
  void operator()(const cv::Mat& src, cv::Mat& dst) const { this->Run(src, dst); }

#ifndef TBB_FORCE_PARALLEL
  // 画像変換
  template <typename Ts, typename Td, typename P>
  void Run(const bool enableParallel, const cv::Mat& src, cv::Mat& dst, P& pa) const {
    if (enableParallel)
      this->Run<Ts, Td, true, P>(src, dst, pa);
    else
      this->Run<Ts, Td, false, P>(src, dst, pa);
  }

  // 画像変換
  template <typename Ts, typename Td>
  void Run(const bool enableParallel, const cv::Mat& src, cv::Mat& dst) const {
    this->Run<Ts, Td, DefaultPartitionerType>(enableParallel, src, dst,
                                              ConvertColor::GetDefaultPartitioner());
  }

  // 画像変換
  template <typename Ts>
  void Run(const bool enableParallel, const cv::Mat& src, cv::Mat& dst) const {
    switch (CV_MAT_DEPTH(dst.type())) {
      case CV_8U:
        this->Run<Ts, std::uint8_t>(enableParallel, src, dst);
        break;
      case CV_8S:
        this->Run<Ts, std::int8_t>(enableParallel, src, dst);
        break;
      case CV_16U:
        this->Run<Ts, std::uint16_t>(enableParallel, src, dst);
        break;
      case CV_16S:
        this->Run<Ts, std::int16_t>(enableParallel, src, dst);
        break;
      case CV_32S:
        this->Run<Ts, std::int32_t>(enableParallel, src, dst);
        break;
      case CV_32F:
        this->Run<Ts, float>(enableParallel, src, dst);
        break;
      case CV_64F:
        this->Run<Ts, double>(enableParallel, src, dst);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }

  // 画像変換
  void Run(const bool enableParallel, const cv::Mat& src, cv::Mat& dst) const {
    switch (CV_MAT_DEPTH(dst.type())) {
      case CV_8U:
        this->Run<std::uint8_t>(enableParallel, src, dst);
        break;
      case CV_8S:
        this->Run<std::int8_t>(enableParallel, src, dst);
        break;
      case CV_16U:
        this->Run<std::uint16_t>(enableParallel, src, dst);
        break;
      case CV_16S:
        this->Run<std::int16_t>(enableParallel, src, dst);
        break;
      case CV_32S:
        this->Run<std::int32_t>(enableParallel, src, dst);
        break;
      case CV_32F:
        this->Run<float>(enableParallel, src, dst);
        break;
      case CV_64F:
        this->Run<double>(enableParallel, src, dst);
        break;
        // 例外処理
      default:
        NoDefault("Unsupported OpenCV matrix type");
    }
  }
#else
  // 画像変換
  template <typename Ts, typename Td, typename P>
  void Run(const bool, const cv::Mat& src, cv::Mat& dst, P& pa) const {
    this->Run<Ts, Td, P>(src, dst, pa);
  }

  // 画像変換
  template <typename Ts, typename Td>
  void Run(const bool enableParallel, const cv::Mat& src, cv::Mat& dst) const {
    this->Run<Ts, Td>(src, dst);
  }

  // 画像変換
  template <typename Ts>
  void Run(const bool, const cv::Mat& src, cv::Mat& dst) const {
    this->Run<Ts>(src, dst);
  }

  // 画像変換
  void Run(const bool, const cv::Mat& src, cv::Mat& dst) const { this->Run(src, dst); }
#endif

  // 画像変換
  void operator()(const bool enableParallel, const cv::Mat& src, cv::Mat& dst) const {
    this->Run(enableParallel, src, dst);
  }

  // 唯一のインスタンスへのアクセス
  static const ConvertColor& Table() {
    static ConvertColor instance;
    return instance;
  }
};
#define CvtBGRtoYUV ConvertColor<imgproc::ConvertColorType::BGRtoYUV>::Table()
#define CvtBGRtoHSV ConvertColor<imgproc::ConvertColorType::BGRtoHSV>::Table()
#define CvtBGRtoLAB ConvertColor<imgproc::ConvertColorType::BGRtoLAB>::Table()
#define CvtYUVtoBGR ConvertColor<imgproc::ConvertColorType::YUVtoBGR>::Table()
#define CvtHSVtoBGR ConvertColor<imgproc::ConvertColorType::HSVtoBGR>::Table()
#define CvtLABtoBGR ConvertColor<imgproc::ConvertColorType::LABtoBGR>::Table()

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
_EXTERN_IMGPROC_ template class _EXPORT_IMGPROC_ ConvertColor<imgproc::ConvertColorType::BGRtoYUV>;

}  // namespace imgproc

#endif
