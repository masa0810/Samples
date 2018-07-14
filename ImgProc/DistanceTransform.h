#pragma once

#include "OpenCvConfig.h"

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <cstddef>

#include <type_traits>
#include <vector>

#include <opencv2/core.hpp>

#include <tbb/tbb.h>

// 警告抑制解除
MSVC_WARNING_POP

namespace imgproc {
namespace detail {

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
  template <typename T>
  static void Transform1D(const int size, const int step, const T* const f, T* const d,
                          int* const i, int* const v, T* const z) {
    const auto* ptrF1 = f + 1;
    auto* ptrV0 = v;
    auto* ptrZ0 = z;
    auto* ptrZ1 = z + 1;

    *ptrV0 = 0;
    *ptrZ0 = std::numeric_limits<T>::lowest();
    *ptrZ1 = std::numeric_limits<T>::max();
    for (auto q = 1; q < size; ++q, ++ptrF1) {
      const auto A = (*ptrF1) + (q * q);
      const auto& v1 = *ptrV0;
      auto s = (A - (f[v1] + (v1 * v1))) / ((q - v1) << 1);
      while (s <= *ptrZ0) {
        --ptrV0;
        --ptrZ0;

        const auto& v2 = *ptrV0;
        s = (A - (f[v2] + (v2 * v2))) / ((q - v2) << 1);
      }
      ++ptrV0;
      ++ptrZ0;

      *ptrV0 = q;
      *ptrZ0 = s;
      *(ptrZ0 + 1) = std::numeric_limits<T>::max();
    }

    auto* ptrD0 = d;
    auto* ptrI0 = i;
    ptrV0 = v;
    for (auto q = 0; q < size; ++q, ptrD0 += step, ++ptrI0) {
      while (*ptrZ1 < q) {
        ++ptrV0;
        ++ptrZ1;
      }
      const auto& refV = *ptrV0;
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
  template <typename T, typename P>
  static void Transform2D(const cv::Size& size, std::vector<T>& data, std::vector<int>& idxX,
                          std::vector<int>& idxY, std::vector<T>& work, std::vector<int>& workV,
                          std::vector<T>& workZ, P& paX, P& paY) {
    // 横方向の距離変換
    tbb::parallel_for(0, size.height,
                      [&](const auto y) {
                        const auto idx = y * size.width;
                        Transform1D(size.width, size.height, &data[idx], &work[y], &idxX[idx],
                                    &workV[idx], &workZ[idx + y]);
                      },
                      paX);

    // 縦方向の距離変換
    tbb::parallel_for(0, size.width,
                      [&](const auto x) {
                        const auto idx = x * size.height;
                        Transform1D(size.height, size.width, &work[idx], &data[x], &idxY[idx],
                                    &workV[idx], &workZ[idx + x]);
                      },
                      paY);
  }

}  // namespace impl

template <typename Tf = float, typename P = const tbb::auto_partitioner>
class DistanceTransform_ {
 public:
  using PartitionerType = P;

 private:
  //! バッファは浮動小数点型のみ
  static_assert(std::is_floating_point<Tf>{}, "template parameter Tf must be floatng point type");

  //
  // メンバ変数
  //

  //! 画像のサイズ
  cv::Size m_size = {};
  //! 計算結果(2乗距離)
  std::vector<Tf> m_sqDist = {};
  //! インデックス(X)
  std::vector<int> m_idxX = {};
  //! インデックス(Y)
  std::vector<int> m_idxY = {};
  //! 二乗距離バッファ
  std::vector<Tf> m_bufSqDist = {};
  //! 計算用バッファ
  std::vector<int> m_bufIdx = {};
  //! 計算用バッファ
  std::vector<Tf> m_bufZ = {};

  //! パーティショナー群
  struct Partitioners {
    PartitionerType Pa = {};
    PartitionerType X = {};
    PartitionerType Y = {};
  };
  //! パーティショナー群
  Partitioners m_partitioners = {};

  /// <summary>
  /// 二値化&距離変換(二乗距離)
  /// </summary>
  /// <param name="src">[in] 入力</param>
  /// <param name="comp">[in] 二値化関数</param>
  /// <returns>変換結果</returns>
  /// <remarks>平方根をしてない結果を返すメソッド</remarks>
  template <typename Ts, typename F>
  const cv::Mat TransformSqDist(const cv::Mat& src, const F& comp) {
    cv::Mat buf(m_size, cv::Type<Tf>(), &m_sqDist[0]);
    tbb::parallel_for(0, src.rows,
                      [&src, &buf, &comp, e = src.cols](const auto y) {
                        const auto* ptrSrc = src.ptr<Ts>(y);
                        auto* ptrBuf = buf.ptr<Tf>(y);
                        for (auto x = 0; x < e; ++x, ++ptrSrc, ++ptrBuf)
                          *ptrBuf = comp(*ptrSrc) ? std::numeric_limits<Tf>::max() : 0;
                      },
                      m_partitioners.Pa);

    // 変換
    detail::Transform2D(m_size, m_sqDist, m_idxX, m_idxY, m_bufSqDist, m_bufIdx, m_bufZ,
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
    tbb::parallel_for(0, dst.rows, [&sqDist, &dst, &conv, e = dst.cols](const auto y) {
      const auto* ptrSqDist = sqDist.ptr<Tf>(y);
      auto* ptrDst = dst.ptr<Td>(y);
      for (auto x = 0; x < e; ++x, ++ptrSqDist, ++ptrDst) *ptrDst = conv(*ptrSqDist);
    });
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

}  // namespace imgproc
