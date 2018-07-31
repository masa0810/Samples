#include "SLIC.h"

#include <Common/CommonFunction.h>

using namespace std;

using namespace cv;
using namespace Eigen;
using namespace tbb;

using namespace commonutility;

#define CONV_TABLE ConvTable()

namespace imgproc {
// 2乗
template <typename T>
T Pow2(T val) {
  return val * val;
}

#pragma region ラベル情報

const bool LabelInfoSLIC::ENABLE_PARALLEL = false;     // 並列化の許可
const bool LabelInfoSLIC::ENABLE_FINAL_UPDATE = true;  // 最終アップデート有効

// コンストラクタ
LabelInfoSLIC::LabelInfoSLIC(const int labelId = 0)
    : LabelId(labelId)  // ラベルID
      ,
      NumOfPix(0)  // ピクセル数
      ,
      MinPos(numeric_limits<int>::max(), numeric_limits<int>::max())  // 最小座標
      ,
      MaxPos(numeric_limits<int>::lowest(), numeric_limits<int>::lowest())  // 最大座標
      ,
      Center(),
      Integration(false),
      m_Sum(0, 0) {}

// +=演算子
const LabelInfoSLIC& LabelInfoSLIC::operator+=(const LabelInfoSLIC& rhs) {
  NumOfPix += rhs.NumOfPix;

  if (MinPos.x > rhs.MinPos.x) MinPos.x = rhs.MinPos.x;
  if (MinPos.y > rhs.MinPos.y) MinPos.y = rhs.MinPos.y;
  if (MaxPos.x < rhs.MaxPos.x) MaxPos.x = rhs.MaxPos.x;
  if (MaxPos.y < rhs.MaxPos.y) MaxPos.y = rhs.MaxPos.y;

  m_Sum += rhs.m_Sum;
  (*this)();

  return *this;
}

// 座標値の追加
void LabelInfoSLIC::Add(const Point& pos, const int) {
  ++NumOfPix;
  if (MinPos.x > pos.x) MinPos.x = pos.x;
  if (MinPos.y > pos.y) MinPos.y = pos.y;
  if (MaxPos.x < pos.x) MaxPos.x = pos.x;
  if (MaxPos.y < pos.y) MaxPos.y = pos.y;

  m_Sum += pos;
}

// ラベル情報の計算
void LabelInfoSLIC::operator()() {
  Center.x = static_cast<double>(m_Sum.x) / NumOfPix;
  Center.y = static_cast<double>(m_Sum.y) / NumOfPix;
}

#pragma endregion

#pragma region RGB to Labテーブル

// RGB表色系からLab表色系への変換(本体)
Vec3d SLIC::RGBtoLAB::Conv(const uchar sR, const uchar sG, const uchar sB) {
  //
  // sRGB->XYZ
  //

  static const Matrix3d toXYZ = (Matrix3d() << 0.4124564, 0.3575761, 0.1804375, 0.2126729,
                                 0.7151522, 0.0721750, 0.0193339, 0.1191920, 0.9503041)
                                    .finished();

  const double R = sR / 255.0;
  const double G = sG / 255.0;
  const double B = sB / 255.0;

  const Vector3d rgb(R <= 0.04045 ? R / 12.92 : pow((R + 0.055) / 1.055, 2.4),
                     G <= 0.04045 ? G / 12.92 : pow((G + 0.055) / 1.055, 2.4),
                     B <= 0.04045 ? B / 12.92 : pow((B + 0.055) / 1.055, 2.4));

  const Vector3d XYZ = toXYZ * rgb;

  //
  // XYZ->LAB
  //

  // actual CIE standard
  constexpr double epsilon = 0.008856;
  constexpr double kappa = 903.3;

  // reference white
  constexpr double Xr = 0.950456;
  constexpr double Yr = 1.0;
  constexpr double Zr = 1.088754;

  const double xr = XYZ.x() / Xr;
  const double yr = XYZ.y() / Yr;
  const double zr = XYZ.z() / Zr;

  const double fx = xr > epsilon ? pow(xr, 1.0 / 3.0) : (kappa * xr + 16.0) / 116.0;
  const double fy = yr > epsilon ? pow(yr, 1.0 / 3.0) : (kappa * yr + 16.0) / 116.0;
  const double fz = zr > epsilon ? pow(zr, 1.0 / 3.0) : (kappa * zr + 16.0) / 116.0;

  return {116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)};
}

// コンストラクタ
SLIC::RGBtoLAB::RGBtoLAB() : m_ColorTable(TABLE_SIZE) {
  tbb_for(blocked_range3d<size_t>(0, VALUE_SIZE, 0, VALUE_SIZE, 0, VALUE_SIZE),
          [&](const blocked_range3d<size_t>& range) {
            for (size_t r = std::begin(range.pages()); r < std::end(range.pages()); ++r) {
              size_t idx = r * IMAGE_SIZE;
              for (size_t g = std::begin(range.rows()); g < std::end(range.rows()); ++g) {
                size_t b = std::begin(range.cols());
                Vec3d* ptr = &m_ColorTable[idx + g * VALUE_SIZE + b];
                for (; b < std::end(range.cols()); ++b, ++ptr)
                  *ptr = Conv(static_cast<uchar>(r), static_cast<uchar>(g), static_cast<uchar>(b));
              }
            }
          });
}

// 変換
const Vec3d& SLIC::RGBtoLAB::operator()(const int val1, const int val2, const int val3) const {
  return m_ColorTable[val3 * IMAGE_SIZE + val2 * VALUE_SIZE + val1];
}

// 変換
const Vec3d& SLIC::RGBtoLAB::operator()(const Vec3b& val) const {
  return (*this)(val[0], val[1], val[2]);
}

// 画像変換
void SLIC::RGBtoLAB::operator()(const Mat& src, Mat& dst) const {
  tbb_for(src.size(), [&](const blocked_range2d<int>& range) {
    for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
      int x = std::begin(range.cols());
      const Vec3b* ptrSrc = src.ptr<Vec3b>(y, x);
      Vec3d* ptrDst = dst.ptr<Vec3d>(y, x);
      for (; x < std::end(range.cols()); ++x, ++ptrSrc, ++ptrDst) *ptrDst = (*this)(*ptrSrc);
    }
  });
}

#pragma endregion

#pragma region Seed情報

SLIC::SeedInfo::Comp::Comp(const SeedInfo& seed)
    : SigLab(0.0, 0.0, 0.0), SigXY(0.0, 0.0), Sum(0), MaxLab(seed.MaxDistLab) {}

SLIC::SeedInfo::Comp::Comp(const Comp& a, const Comp& b)
    : SigLab(a.SigLab + b.SigLab),
      SigXY(a.SigXY + b.SigXY),
      Sum(a.Sum + b.Sum),
      MaxLab(max(a.MaxLab, b.MaxLab)) {}

SLIC::SeedInfo::SeedInfo(const Point2d& xy, const Vec3d& lab, const double initDist)
    : XY(xy),
      Lab(lab),
      MinPos(),
      MaxPos(),
      Integration(-1),
      SigmaLab(),
      SigmaXY(),
      ClusterSize(),
      MaxDistLab(Pow2(initDist)) {}

void SLIC::SeedInfo::Set(const Comp& comp) {
  SigmaLab = comp.SigLab;
  SigmaXY = comp.SigXY;
  ClusterSize = comp.Sum;
  MaxDistLab = comp.MaxLab;
}

#pragma endregion

// 近傍テーブル
const array<Point, 8> SLIC::D8 = {Point(1, 0),  Point(1, 1),   Point(0, 1),  Point(-1, 1),
                                  Point(-1, 0), Point(-1, -1), Point(0, -1), Point(1, -1)};

// 変換テーブル
const SLIC::RGBtoLAB& SLIC::ConvTable() {
  static const RGBtoLAB instance;  // 変換テーブル
  return instance;
}

// RGB表色系系からLab表色系への変換
void SLIC::RgbToLabConversion(const Mat& src, const Mat& mask, Mat& lab) {
  tbb_for(src.size(), [&](const blocked_range2d<int>& range) {
    for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
      int x = std::begin(range.cols());
      const Vec3b* ptrSrc = src.ptr<Vec3b>(y, x);
      const uchar* ptrMask = mask.ptr<uchar>(y, x);
      Vec3d* ptrLab = lab.ptr<Vec3d>(y, x);
      for (; x < std::end(range.cols()); ++x, ++ptrSrc, ++ptrMask, ++ptrLab)
        *ptrLab = *ptrMask ? CONV_TABLE(*ptrSrc) : Vec3d(0.0, 0.0, 0.0);
    }
  });
}

// Lab空間での入力画像の勾配を求める
void SLIC::DetectLabEdge(const Mat& src, Mat& dst) {
  fill(dst.ptr<double>(), dst.ptr<double>(1, 0), 0.0);  // 上1ピクセルの初期化
  tbb_for(1, src.rows - 1, [&](const int y) {
    double* ptr = dst.ptr<double>(y, 0);
    const double* const ptrEnd = dst.ptr<double>(y, dst.cols - 1);
    const Vec3d* ptrL = src.ptr<Vec3d>(y, 0);
    const Vec3d* ptrR = ptrL + 2;
    const Vec3d* ptrU = ptrL - src.cols + 1;
    const Vec3d* ptrB = ptrL + src.cols + 1;

    *ptr = 0.0;  // 左1ピクセルの初期化
    for (++ptr; ptr < ptrEnd; ++ptr, ++ptrL, ++ptrR, ++ptrU, ++ptrB) {
      *ptr = Pow2((*ptrL)[0] - (*ptrR)[0]) + Pow2((*ptrL)[1] - (*ptrR)[1]) +
             Pow2((*ptrL)[2] - (*ptrR)[2]) + Pow2((*ptrU)[0] - (*ptrB)[0]) +
             Pow2((*ptrU)[1] - (*ptrB)[1]) + Pow2((*ptrU)[2] - (*ptrB)[2]);
    }
    *ptr = 0.0;  // 右1ピクセルの初期化
  });
  fill_n(dst.ptr<double>(dst.rows - 1, 0), dst.cols, 0.0);  // 下1ピクセルの初期化
}

// シード位置を微調整
void SLIC::PerturbSeed(const Size& size, const vector<Vec3d>& labs, const vector<double>& edge,
                       const Mat& mask, SeedInfo& seed) {
  // 最小エッジ探索用構造体
  struct MinEdgeSearch {
    double MinEdge;
    int Idx;
    Point Pos;
    explicit MinEdgeSearch(double minEdge) : MinEdge(minEdge), Idx(-1), Pos() {}
  };

  // 近傍と比較する
  const Point orig = seed.XY;
  MinEdgeSearch ret = serial_reduce /*tbb_reduce*/ (
      blocked_range<size_t>(0, D8.size()), MinEdgeSearch(edge[orig.y * size.width + orig.x]),
      [&](const blocked_range<size_t>& range, MinEdgeSearch val) -> MinEdgeSearch {
        for (size_t i = std::begin(range); i < std::end(range); ++i) {
          const Point pos = orig + D8[i];
          const int idx = pos.y * size.width + pos.x;
          if (AreaCheck(pos, size) && mask.at<uchar>(pos)) {
            const double& edgeVal = edge[idx];
            if (edgeVal < val.MinEdge) {
              val.MinEdge = edgeVal;
              val.Idx = idx;
              val.Pos = pos;
            }
          }
        }
        return val;
      },
      [](const MinEdgeSearch& a, const MinEdgeSearch& b) -> MinEdgeSearch {
        return (a.MinEdge < b.MinEdge) ? a : b;
      });

  // 変更があった場合、L*a*b*と座標を更新
  if (ret.Idx >= 0) {
    seed.XY = ret.Pos;
    seed.Lab = labs[ret.Idx];
  }
}

// 小さいセグメントを隣接するセグメントに統合する
void SLIC::EnforceLabelConnectivity(const Mat& mask, const int area, const int numOfSp,
                                    LabelingType& labeling, vector<LabelInfoSLIC>& labelInfos,
                                    vector<int>& labels) {
  struct MinClusterSearch {
    int Count;
    int Idx;

    MinClusterSearch() : Count(numeric_limits<int>::max()), Idx(-1) {}
  };

  const Size& size = mask.size();
  const int supsZper4 = (area / numOfSp) >> 2;

  Mat seedLabel(size, CV_32SC1, static_cast<void*>(&labels[0]));
  //labeling(seedLabel);
  labeling.CompEqual(seedLabel);
  labeling.CreateLabelInfo<LabelInfoSLIC>(seedLabel, labelInfos);

  tbb_for(size, [&](const blocked_range2d<int>& range) {
    for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
      int x = std::begin(range.cols());
      const uchar* ptrMask = mask.ptr<uchar>(y, x);
      const SLIC::LabelingType::ImgType* ptrLabeling =
          labeling.GetLabelingImg().ptr<SLIC::LabelingType::ImgType>(y, x);
      int* ptrLabel = seedLabel.ptr<int>(y, x);
      for (; x < std::end(range.cols()); ++x, ++ptrMask, ++ptrLabeling, ++ptrLabel) {
        *ptrLabel = (*ptrMask) ? *ptrLabeling : -1;
      }
    }
  });

  int numOfSeed = static_cast<int>(labelInfos.size());
  for (int i = 0; i < static_cast<int>(labelInfos.size()); ++i) {
    LabelInfoSLIC& label = labelInfos[i];

    if (label.NumOfPix < supsZper4) {
      const Point start(max(label.MinPos.x - 1, 0), max(label.MinPos.y - 1, 0));
      const Point end(min(label.MaxPos.x + 2, size.width), min(label.MaxPos.y + 2, size.height));
      MinClusterSearch ret = serial_reduce /*tbb_reduce*/ (
          start, end, MinClusterSearch(),
          [&](const blocked_range2d<int>& range, MinClusterSearch val) -> MinClusterSearch {
            for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
              int x = std::begin(range.cols());
              const int* ptrLabel = seedLabel.ptr<int>(y, x);
              for (; x < std::end(range.cols()); ++x, ++ptrLabel) {
                if (*ptrLabel != i && *ptrLabel >= 0) {
                  const LabelInfoSLIC& label2 = labelInfos[*ptrLabel];
                  if ((label2.NumOfPix >= supsZper4) && (val.Count > label2.NumOfPix)) {
                    val.Count = label2.NumOfPix;
                    val.Idx = *ptrLabel;
                  }
                }
              }
            }
            return val;
          },
          [](const MinClusterSearch& a, const MinClusterSearch& b) -> MinClusterSearch {
            return a.Count < b.Count ? a : b;
          });

      if (ret.Idx >= 0) {
        --numOfSeed;
        label.Integration = true;
        labelInfos[ret.Idx] += label;

        tbb_for(blocked_range2d<int>(label.MinPos.y, label.MaxPos.y + 1, label.MinPos.x,
                                     label.MaxPos.x + 1),
                [&](const blocked_range2d<int>& range) {
                  for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
                    int x = std::begin(range.cols());
                    int* ptrLabel = seedLabel.ptr<int>(y, x);
                    for (; x < std::end(range.cols()); ++x, ++ptrLabel) {
                      if (*ptrLabel == i) *ptrLabel = ret.Idx;
                    }
                  }
                });
      }
    }
  }

  tbb_sort(labelInfos, [](const LabelInfoSLIC& a, const LabelInfoSLIC& b) {
    return (a.Integration ^ b.Integration) ? b.Integration : a.LabelId < b.LabelId;
  });

  labelInfos.resize(numOfSeed);
}

// スーパーピクセル生成
void SLIC::PerformSuperpixel(const Mat& mask, const int area, const int numOfSp,
                             const bool flagPerturbSeed, const int numOfK, const double initDist) {
  //
  // Seed数計算
  //
  double step = sqrt(static_cast<double>(area) / numOfSp);
  const Point strips(cvFloor(m_ImgSize.width / step), cvFloor(m_ImgSize.height / step));
  const Point2d stepXY(static_cast<double>(m_ImgSize.width / strips.x),
                       static_cast<double>(m_ImgSize.height) / strips.y);
  const Point2d offset = stepXY * 0.5;

  //
  // Seed位置セット
  //
  m_Seeds.clear();
  const Size maxSize(m_ImgSize.width - 1, m_ImgSize.height - 1);
  tbb_for(blocked_range2d<int>(0, strips.y, 0, strips.x), [&](const blocked_range2d<int>& range) {
    for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
      const double Y = min<double>(y * stepXY.y + offset.y, maxSize.height);
      const int _Y = cvRound(Y);
      //const bool yf = y & 0x1;    // hex grid

      for (int x = std::begin(range.cols()); x < std::end(range.cols()); ++x) {
        //const double X = x * stepXY.x + (offset.x * (yf ^ (x & 0x1)));    // hex grid
        const double X = min<double>(x * stepXY.x + offset.x, maxSize.width);  // grid
        const int _X = cvRound(X);
        const int idx = _Y * m_ImgSize.width + _X;

        if (mask.at<uchar>(_Y, _X)) {
          // Seed作成
          SeedInfo seed(Point2d(X, Y), m_Labs[idx], initDist);

          // Seed位置微調整
          if (flagPerturbSeed) PerturbSeed(m_ImgSize, m_Labs, m_LabEdge, mask, seed);

          // Seedセット
          m_Seeds.emplace_back(seed);
        }
      }
    }
  });

  //
  // ソート
  // ※ シードが並列にプッシュされるので、順序を並列化に依存しないようにする
  //
  tbb_sort(m_Seeds, [&](const SeedInfo& a, const SeedInfo& b) {
    return (a.XY.y == b.XY.y) ? (a.XY.x < b.XY.x) : (a.XY.y < b.XY.y);
  });

  //
  // スーパーピクセル生成
  //
  const Point2d stepOffset((stepXY.x < 10.0) ? stepXY.x * 1.5 : stepXY.x,
                           (stepXY.y < 10.0) ? stepXY.y * 1.5 : stepXY.y);
  const double waitXY = 1.0 / Pow2(step);
  fill(std::begin(m_Labels), std::end(m_Labels), -1);  // ラベル番号初期化
  for (int n = 0; n < numOfK; ++n) {
    fill(std::begin(m_DistVecs), std::end(m_DistVecs), numeric_limits<double>::max());  // 最小距離初期化

    // 各Seedからの距離計算
    for (int i = 0; i < static_cast<int>(m_Seeds.size()); ++i) {
      SeedInfo& seed = m_Seeds[i];

      seed.MinPos.x = max(cvFloor(seed.XY.x - stepOffset.x), 0);
      seed.MinPos.y = max(cvFloor(seed.XY.y - stepOffset.y), 0);
      seed.MaxPos.x = min(cvCeil(seed.XY.x + stepOffset.x) + 1, m_ImgSize.width);
      seed.MaxPos.y = min(cvCeil(seed.XY.y + stepOffset.y) + 1, m_ImgSize.height);
      tbb_for(seed.MinPos, seed.MaxPos, [&](const blocked_range2d<int>& range) {
        for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
          int x = std::begin(range.cols());
          const int idx = y * m_ImgSize.width + x;
          const uchar* ptrMask = mask.ptr<uchar>(y, x);
          const Vec3d* ptrLabs = &m_Labs[idx];
          double* ptrDistLab = &m_DistLabs[idx];
          double* ptrDistVec = &m_DistVecs[idx];
          int* ptrLabel = &m_Labels[idx];
          for (; x < std::end(range.cols());
               ++x, ++ptrMask, ++ptrLabs, ++ptrDistLab, ++ptrDistVec, ++ptrLabel) {
            if (*ptrMask) {
              const Vec3d subLab = *ptrLabs - seed.Lab;
              const double distLab = Pow2(subLab[0]) + Pow2(subLab[1]) + Pow2(subLab[2]);
              const double distXY = Pow2(x - seed.XY.x) + Pow2(y - seed.XY.y);
              const double dist = distLab / seed.MaxDistLab + distXY * waitXY;  // サイズが一定になるような評価関数

              // 最小距離比較
              if (dist < *ptrDistVec) {
                *ptrDistVec = dist;     // 最小距離保存
                *ptrDistLab = distLab;  // L*a*b*の値保存
                *ptrLabel = i;          // ラベル設定
              }
            }
          }
        }
      });
    }

    // Seedの更新処理
    tbb_for(0, static_cast<int>(m_Seeds.size()), [&](int i) {
      SLIC::SeedInfo& seed = m_Seeds[i];

      seed.Set(serial_reduce /*tbb_reduce*/ (
          seed.MinPos, seed.MaxPos, SLIC::SeedInfo::Comp(seed),
          [&](const blocked_range2d<int>& range, SLIC::SeedInfo::Comp val) -> SLIC::SeedInfo::Comp {
            for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
              int x = std::begin(range.cols());
              const int idx = y * m_ImgSize.width + x;
              const Vec3d* ptrLabs = &m_Labs[idx];
              const double* ptrDistLab = &m_DistLabs[idx];
              const int* ptrLabel = &m_Labels[idx];
              for (; x < std::end(range.cols()); ++x, ++ptrLabs, ++ptrDistLab, ++ptrLabel) {
                if (*ptrLabel == i) {
                  if (val.MaxLab < *ptrDistLab) val.MaxLab = *ptrDistLab;

                  val.SigLab += *ptrLabs;
                  val.SigXY += Point2d(x, y);
                  ++val.Sum;
                }
              }
            }
            return val;
          },
          [](const SLIC::SeedInfo::Comp& a, const SLIC::SeedInfo::Comp& b) -> SLIC::SeedInfo::Comp {
            return SLIC::SeedInfo::Comp(a, b);
          }));

      const double invClusterSize = (seed.ClusterSize > 0) ? 1.0 / seed.ClusterSize : 1.0;
      seed.Lab = seed.SigmaLab * invClusterSize;
      seed.XY = seed.SigmaXY * invClusterSize;
    });
  }
}

// 実行
void SLIC::Run(const cv::Mat& src, const cv::Mat& mask, const int numOfSp, const int area,
               const bool flagPerturbSeed, const int numOfK, const double initDist) {
  // バッファサイズチェック
  if (m_ImgSize != src.size()) {
    m_ImgSize = src.size();
    const int newSize = m_ImgSize.area();

    m_Labels.resize(newSize);
    m_Labs.resize(newSize);
    m_LabEdge.resize(newSize);
    m_DistLabs.resize(newSize);
    m_DistVecs.resize(newSize);
    m_Labeling = LabelingType(m_ImgSize);
    m_BufDbg.resize(newSize);
  }

  // RGB to L*a*b*
  Mat lab(m_ImgSize, CV_64FC3, static_cast<void*>(&m_Labs[0]));
  RgbToLabConversion(src, mask, lab);

  // Lab空間でのエッジ検出
  // ※ エッジ上にシードがセットされるのを防ぐため
  if (flagPerturbSeed) {
    Mat labEdge(m_ImgSize, CV_64FC1, static_cast<void*>(&m_LabEdge[0]));
    DetectLabEdge(lab, labEdge);
  }

  // スーパーピクセル生成
  PerformSuperpixel(mask, area, numOfSp, flagPerturbSeed, numOfK, initDist);

  // 小さいセグメントを隣接セグメントに統合する
  if (m_Seeds.size())
    EnforceLabelConnectivity(mask, area, numOfSp, m_Labeling, m_LabelInfos, m_Labels);
  else
    m_LabelInfos.clear();
}

// コンストラクタ
SLIC::SLIC(const Size& size)
    : m_ImgSize(size),
      m_NumOfLabel(0),
      m_Labels(size.area()),
      m_Labs(size.area()),
      m_LabEdge(size.area()),
      m_DistLabs(size.area()),
      m_DistVecs(size.area()),
      m_Seeds(),
      m_Labeling(size),
      m_LabelInfos(),
      m_DummyMaskData(m_ImgSize.area(), 255),
      m_BufDbg(m_ImgSize.area()) {
  CONV_TABLE;
}

// スーパーピクセルセグメンテーション実行(マスク対応)
void SLIC::operator()(const Mat& src, const Mat& mask, const int numOfSp,
                      const bool flagPerturbSeed, const int numOfK, const double initDist) {
  // マスク領域カウント
  std::atomic<int> area;
  area = 0;
  tbb_for(mask.size(), [&](const blocked_range2d<int>& range) {
    for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
      int x = std::begin(range.cols());
      const uchar* ptrMask = mask.ptr<uchar>(y, x);
      for (; x < std::end(range.cols()); ++x, ++ptrMask) area += (*ptrMask != 0);
    }
  });

  Run(src, mask, numOfSp, area, flagPerturbSeed, numOfK, initDist);
}

// スーパーピクセルセグメンテーション実行
void SLIC::operator()(const Mat& src, const int numOfSp, const bool flagPerturbSeed,
                      const int numOfK, const double initDist) {
  // バッファサイズチェック
  if (m_ImgSize != src.size()) m_DummyMaskData.assign(src.size().area(), 255);
  Run(src, Mat(src.size(), CV_8UC1, static_cast<void*>(&m_DummyMaskData[0])), numOfSp,
      m_ImgSize.area(), flagPerturbSeed, numOfK, initDist);
}

// スーパーピクセルセグメンテーション実行(マスク対応)
void SLIC::FixedInterval(const cv::Mat& src, const cv::Mat& mask, const int numOfSp,
                         const bool flagPerturbSeed, const int numOfK, const double initDist) {
  Run(src, mask, numOfSp, m_ImgSize.area(), flagPerturbSeed, numOfK, initDist);
}

// スーパーピクセルセグメンテーション実行
void SLIC::PixStep(const cv::Mat& src, const cv::Mat& mask, const int step,
                   const bool flagPerturbSeed, const int numOfK, const double initDist) {
  Run(src, mask, m_ImgSize.area() / Pow2(step), m_ImgSize.area(), flagPerturbSeed, numOfK,
      initDist);
}

// スーパーピクセルセグメンテーション実行
void SLIC::PixStep(const cv::Mat& src, const int step, const bool flagPerturbSeed, const int numOfK,
                   const double initDist) {
  (*this)(src, m_ImgSize.area() / Pow2(step), flagPerturbSeed, numOfK, initDist);
}

// 画像サイズ
const cv::Size& SLIC::GetSize() const { return m_ImgSize; }

// 最終的なラベル数取得
size_t SLIC::GetNumOfLabel() const { return m_LabelInfos.size(); }

// ラベル情報取得
const vector<LabelInfoSLIC>& SLIC::GetLabelInfo() const { return m_LabelInfos; }

// ラベル画像取得
const cv::Mat SLIC::GetLabelImg() const {
  const Mat labelImg(m_ImgSize, CV_32SC1, static_cast<void*>(const_cast<int*>(&m_Labels[0])));
  return labelImg;
}

// デバッグ画像取得
void SLIC::CreateDebugImage(Mat& dst) {
  const Mat seedLabel(m_ImgSize, CV_32SC1, static_cast<void*>(&m_Labels[0]));
  Mat buf(m_ImgSize, CV_8UC1, static_cast<void*>(&m_BufDbg[0]));
  buf = 0;

  auto CalcIdx = [](const Point& p, const int s) -> int { return p.y * s + p.x; };

  const int seedLabelStep = static_cast<int>(seedLabel.step[0] / seedLabel.step[1]);
  const array<int, 8> idxs = {CalcIdx(D8[0], seedLabelStep), CalcIdx(D8[1], seedLabelStep),
                              CalcIdx(D8[2], seedLabelStep), CalcIdx(D8[3], seedLabelStep),
                              CalcIdx(D8[4], seedLabelStep), CalcIdx(D8[5], seedLabelStep),
                              CalcIdx(D8[6], seedLabelStep), CalcIdx(D8[7], seedLabelStep)};

  const int dstStep = static_cast<int>(dst.step[0] / dst.step[1]);
  const array<int, 8> dstIdxs = {CalcIdx(D8[0], dstStep), CalcIdx(D8[1], dstStep),
                                 CalcIdx(D8[2], dstStep), CalcIdx(D8[3], dstStep),
                                 CalcIdx(D8[4], dstStep), CalcIdx(D8[5], dstStep),
                                 CalcIdx(D8[6], dstStep), CalcIdx(D8[7], dstStep)};

  auto GetParam = [&](const int x, const int y) -> pair<int, int> {
    switch ((x == m_ImgSize.width - 1) |        // Right
            (y == m_ImgSize.height - 1) << 1 |  // Bottom
            (x == 0) << 2 |                     // Left
            (y == 0) << 3)                      // Top
    {
      case 8:  // Top
        return {5, 0};
      case 1:  // Right
        return {5, 2};
      case 2:  // Bottom
        return {5, 4};
      case 4:  // Left
        return {5, 6};
      case 12:  // Left Top
        return {3, 0};
      case 9:  // Right Top
        return {3, 2};
      case 3:  // Right Bottom
        return {3, 4};
      case 6:  // Left Bottom
        return {3, 6};
      default:
        return {static_cast<int>(idxs.size()), 5};
    }
  };

  tbb_for_each(m_LabelInfos, [&](const LabelInfoSLIC& label) {
    tbb_for(blocked_range2d<int>(label.MinPos.y, label.MaxPos.y + 1, label.MinPos.x,
                                 label.MaxPos.x + 1),
            [&](const blocked_range2d<int>& range) {
              for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
                int x = std::begin(range.cols());
                const int* ptr = seedLabel.ptr<int>(y, x);
                Vec3b* ptrDst = dst.ptr<Vec3b>(y, x);
                uchar* ptrBuf = buf.ptr<uchar>(y, x);
                for (; x < std::end(range.cols()); ++x, ++ptr, ++ptrDst, ++ptrBuf) {
                  if (*ptr == label.LabelId) {
                    pair<int, int> param = GetParam(x, y);
                    for (int i = 0; i < param.first; ++i) {
                      const int idx = (i + param.second) & 0x07;
                      if (*(ptr + idxs[idx]) != label.LabelId) {
                        *ptrDst = Vec3b(255, 255, 255);
                        *ptrBuf = 255;

                        for (int j = 0; j < param.first; ++j) {
                          int idx2 = (j + param.second) & 0x07;
                          const int& idxVal = idxs[idx2];
                          uchar* ptrTmpBuf = ptrBuf + idxVal;
                          if (*(ptr + idxVal) == label.LabelId && *ptrTmpBuf == 0) {
                            *(ptrDst + dstIdxs[idx2]) = Vec3b(0, 0, 0);
                            *ptrTmpBuf = 255;
                          }
                        }
                      }
                    }
                  }
                }
              }
            });
  });
}

// デバッグ画像取得
void SLIC::CreateDebugLabelImage(cv::Mat& dst, bool flag) {
  LabelInfoSLIC::CreateDbgImg<int>(dst, Mat(m_ImgSize, CV_32SC1, static_cast<void*>(&m_Labels[0])),
                                   m_LabelInfos, flag);
}
}  // namespace imgproc
