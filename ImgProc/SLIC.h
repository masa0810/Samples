#pragma once

#include "Labeling.h"

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

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
// ラベル情報
struct _EXPORT_IMGPROC_ LabelInfoSLIC : boost::addable<LabelInfoSLIC> {
  static const bool ENABLE_PARALLEL;      // 並列化の許可
  static const bool ENABLE_FINAL_UPDATE;  // 最終アップデート有効
  int LabelId;                            // ラベルID
  int NumOfPix;                           // ピクセル数
  cv::Point MinPos;                       // 最小座標
  cv::Point MaxPos;                       // 最大座標
  cv::Point2d Center;                     // 中心座標

  // コンストラクタ
  explicit LabelInfoSLIC(const int labelId);

  // +=演算子
  const LabelInfoSLIC& operator+=(const LabelInfoSLIC& rhs);

 protected:
  friend struct Labeling<int>;
  friend class SLIC;

  bool Integration;  // 結合
  cv::Point m_Sum;

  // 座標値の追加
  void Add(const cv::Point& pos, const int);

  // ラベル情報の計算
  void operator()();

  // デバッグ画像作成
  template <typename ImgType>
  static void CreateDbgImg(cv::Mat& dst, const cv::Mat& labelingImg,
                           const std::vector<LabelInfoSLIC>& LabelInfoSLICs,
                           const bool flag = false) {
    using namespace std;
    using namespace cv;
    using namespace tbb;
    using namespace commonutility;

    tbb_for(dst.size(), [&](const blocked_range2d<int>& range) {
      for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
        int x = std::begin(range.cols());
        const int* ptr = labelingImg.ptr<int>(y, x);
        Vec3b* ptrDst = dst.ptr<Vec3b>(y, x);
        for (; x < std::end(range.cols()); ++x, ++ptr, ++ptrDst) {
          *ptrDst = *ptr < 0 ? Vec3b(0, 0, 0) : ColorTable[*ptr % 256];
        }
      }
    });

    if (flag) {
      tbb_for_each(LabelInfoSLICs, [&](const LabelInfoSLIC& itm) {
        circle(dst, itm.Center, 2, Scalar(0, 0, 0));
        rectangle(dst, itm.MinPos, itm.MaxPos, Scalar(0, 0, 0));
      });
    }
  }
};

class _EXPORT_IMGPROC_ SLIC {
  // RGB表色系からLab表色系への変換クラス
  class RGBtoLAB {
    static constexpr size_t VALUE_SIZE = 256;
    static constexpr size_t IMAGE_SIZE = VALUE_SIZE * VALUE_SIZE;
    static constexpr size_t TABLE_SIZE = VALUE_SIZE * VALUE_SIZE * VALUE_SIZE;

    std::vector<cv::Vec3d> m_ColorTable;  // 色テーブル

    // RGB表色系からLab表色系への変換(本体)
    static cv::Vec3d Conv(const uchar sR, const uchar sG, const uchar sB);

   public:
    // コンストラクタ
    RGBtoLAB();

    // 変換
    const cv::Vec3d& operator()(const int val1, const int val2, const int val3) const;

    // 変換
    const cv::Vec3d& operator()(const cv::Vec3b& val) const;

    // 画像変換
    void operator()(const cv::Mat& src, cv::Mat& dst) const;
  };

  // Seed情報
  struct SeedInfo {
    cv::Point2d XY;  // 画像位置
    cv::Vec3d Lab;   // L*a*b*

    cv::Point MinPos;  // 最小位置
    cv::Point MaxPos;  // 最大位置

    int Integration;

    // 計算用変数
    cv::Point2d SigmaXY;
    cv::Vec3d SigmaLab;
    int ClusterSize;
    double MaxDistLab;

    // 比較用構造体
    struct Comp {
      cv::Vec3d SigLab;
      cv::Point2d SigXY;
      int Sum;
      double MaxLab;

      explicit Comp(const SeedInfo& seed);

      Comp(const Comp& a, const Comp& b);
    };

    SeedInfo(const cv::Point2d& xy, const cv::Vec3d& lab, const double initDist);

    void Set(const Comp& comp);
  };

  static const std::array<cv::Point, 8> D8;

  cv::Size m_ImgSize;  // 画像サイズ
  int m_NumOfLabel;    // 最終的なラベル数

  // 計算用内部情報
  std::vector<int> m_Labels;                 // ラベル情報
  std::vector<cv::Vec3d> m_Labs;             // L*a*b*色空間での画像情報
  std::vector<double> m_LabEdge;             // L*a*b*のエッジ情報
  std::vector<double> m_DistLabs;            // L*a*b*色空間での距離
  std::vector<double> m_DistVecs;            // 上記2種の複合距離
  tbb::concurrent_vector<SeedInfo> m_Seeds;  // 各Seed情報

  //using LabelingType = LabelingFast<std::equal_to<int>, int>;
  using LabelingType = Labeling<int>;

  LabelingType m_Labeling;
  std::vector<LabelInfoSLIC> m_LabelInfos;

  std::vector<uchar> m_DummyMaskData;  // ダミーマスクデータ
  std::vector<uchar> m_BufDbg;         // デバッグ画像作成用

  // 変換テーブル
  static const RGBtoLAB& ConvTable();

  // RGB表色系系からLab表色系への変換
  static void RgbToLabConversion(const cv::Mat& src, const cv::Mat& mask, cv::Mat& lab);

  // Lab空間での入力画像の勾配を求める
  static void DetectLabEdge(const cv::Mat& src, cv::Mat& dst);

  // シード位置を微調整
  static void PerturbSeed(const cv::Size& size, const std::vector<cv::Vec3d>& labs,
                          const std::vector<double>& edge, const cv::Mat& mask, SeedInfo& seed);

  // 小さいセグメントを隣接するセグメントに統合する(マスク対応)
  static void EnforceLabelConnectivity(const cv::Mat& mask, const int area, const int numOfSp,
                                       LabelingType& labeling,
                                       std::vector<LabelInfoSLIC>& labelInfos,
                                       std::vector<int>& labels);

  // スーパーピクセル生成
  void PerformSuperpixel(const cv::Mat& mask, const int area, const int numOfSp,
                         const bool flagPerturbSeed, const int numOfK, const double initDist);

  // 実行
  void Run(const cv::Mat& src, const cv::Mat& mask, const int numOfSp, const int area,
           const bool flagPerturbSeed, const int numOfK, const double initDist);

 public:
  using LabelImgType = int;

  // コンストラクタ
  explicit SLIC(const cv::Size& size = cv::Size(0, 0));

  // スーパーピクセルセグメンテーション実行(マスク対応)
  void operator()(const cv::Mat& src, const cv::Mat& mask, const int numOfSp = 200,
                  const bool flagPerturbSeed = false, const int numOfK = 10,
                  const double initDist = 10.0);

  // スーパーピクセルセグメンテーション実行
  void operator()(const cv::Mat& src, const int numOfSp = 200, const bool flagPerturbSeed = false,
                  const int numOfK = 10, const double initDist = 10.0);

  // スーパーピクセルセグメンテーション実行(マスク対応)
  void FixedInterval(const cv::Mat& src, const cv::Mat& mask, const int numOfSp = 200,
                     const bool flagPerturbSeed = false, const int numOfK = 10,
                     const double initDist = 10.0);

  // スーパーピクセルセグメンテーション実行
  void PixStep(const cv::Mat& src, const cv::Mat& mask, const int step = 10,
               const bool flagPerturbSeed = false, const int numOfK = 10,
               const double initDist = 10.0);

  // スーパーピクセルセグメンテーション実行
  void PixStep(const cv::Mat& src, const int step = 10, const bool flagPerturbSeed = false,
               const int numOfK = 10, const double initDist = 10.0);

  // 画像サイズ
  const cv::Size& GetSize() const;

  // 最終的なラベル数取得
  size_t GetNumOfLabel() const;

  // ラベル情報取得
  const std::vector<LabelInfoSLIC>& GetLabelInfo() const;

  // ラベル画像取得
  const cv::Mat GetLabelImg() const;

  // デバッグ画像取得
  void CreateDebugImage(cv::Mat& dst);

  // デバッグ画像取得
  void CreateDebugLabelImage(cv::Mat& dst, bool flag = false);
};
}  // namespace imgproc
