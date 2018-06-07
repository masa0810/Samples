#pragma once

#include <Common/CommonDef.h>
#include <Common/CommonTable.h>
#include <Common/OpenCvConfig.h>
#include <Common/TbbConfig.h>

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <atomic>
#include <iterator>
#include <mutex>

#include <boost/operators.hpp>

#include <opencv2/imgproc.hpp>

#include <tbb/flow_graph.h>

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

MSVC_WARNING_PUSH
MSVC_WARNING_DISABLE(127)  // 定数条件式警告対応

namespace imgproc {
namespace labeling {
// 絶対値比較
class _EXPORT_IMGPROC_ AbsDif {
  int m_DifVal;

 public:
  explicit AbsDif(int difVal);

  template <typename T>
  bool operator()(const T& val, const T& valPrv) const {
    return std::abs(val - valPrv) <= m_DifVal;
  }
};
}  // namespace labeling

namespace detail {
// ラベル管理(スレッドセーフ)
template <size_t MAX_NUM_LABEL>
class LabelManager {
  std::vector<int> m_Lut;     // ラベル番号統合用ルックアップテーブル
  std::vector<int> m_NumLut;  // ラベル番号統合用ルックアップテーブル

  std::mutex m_LutMutex;     // ルックアップテーブル用ミューテックス
  std::atomic<int> m_Count;  // ラベル数

  // ラベルカウント
  int CountUp() { return ++m_Count; }

  // 初期化
  void Clear() {
    m_Count = 0;
    std::fill(std::begin(m_Lut), std::end(m_Lut), 0);
  }

  // 最終ラベル番号取得(再帰呼び出し)
  int& GetLastChain(int idx) {
    int& lut = m_Lut[idx];     // 現在のラベル
    int& refLut = m_Lut[lut];  // 一つ前の連結ラベル
    if (lut != refLut)         // 比較
    {
      int& lastLut = GetLastChain(refLut);  // 連結の最初のラベル
      lut = lastLut;                        // 一つ後のラベル番号を書き換え
      return lastLut;                       // 最終ラベル番号を返す
    } else
      return refLut;  // すでに最終ラベル番号なので、何もせずに返す。
  }

 public:
  // コンストラクタ
  LabelManager() : m_Lut(MAX_NUM_LABEL), m_NumLut(MAX_NUM_LABEL), m_LutMutex(), m_Count() {
    m_Count = 0;
  }

  // コピーコンストラクタ
  LabelManager(const LabelManager& rhs)
      : m_Lut(rhs.m_Lut), m_NumLut(rhs.m_NumLut), m_LutMutex(), m_Count() {
    m_Count = static_cast<int>(rhs.m_Count);
  }

  // スワップ
  void swap(LabelManager& rhs) {
    std::swap(m_Lut, rhs.m_Lut);
    std::swap(m_NumLut, rhs.m_NumLut);

    int count = m_Count;
    int rhsCount = rhs.m_Count;
    m_Count = rhsCount;
    rhs.m_Count = count;
  }

  // 代入演算子
  LabelManager& operator=(const LabelManager& rhs) {
    LabelManager tmp = rhs;  // コピーコンストラクタによってコピーを作成
    this->swap(tmp);         // 作成したコピーと自身の値入れ替え
    return *this;            // 入れ替え後の自身を返す
  }

  // intへのキャスト
  operator int() const { return m_Count; }

  // ラベル更新
  // ※ ラベル番号を遡って修正
  void operator()(int idxX, int idxY) {
    std::lock_guard<std::mutex> lock(m_LutMutex);

    // ラベル番号に対応する最終番号取得
    int& idxA = GetLastChain(idxX);
    int& idxB = GetLastChain(idxY);

    // より小さいラベル番号に変更
    if (idxA != idxB) {
      if (idxA < idxB)
        idxB = idxA;
      else
        idxA = idxB;
    }
  }

  // インクリメント
  int operator++() {
    int idx = CountUp();  // ラベル番号をカウントアップ
    m_Lut[idx] = idx;     // ラベル番号追加
    return idx;           // ラベル番号を返す
  }

  // 初期化
  LabelManager& operator=(int n) {
    if (!n) Clear();

    return *this;
  }

  // ルックアップテーブルの値取得
  const int& operator[](int idx) const { return m_Lut[idx]; }

  // ラベル統合
  void IntegratedLabel() {
    using namespace tbb;

    const int itEnd = m_Count + 1;
    m_Count = 1;
    for (int i = 1; i < itEnd; ++i) {
      int& lut = m_Lut[i];
      int& refLut = m_Lut[lut];
      if (lut != refLut) lut = GetLastChain(refLut);  // 再帰的にラベル番号を探索

      if (i == lut) m_NumLut[i] = m_Count++;
    }

    tbb_for_each(&m_Lut[1], &m_Lut[itEnd], [&](int& lut) { lut = m_NumLut[lut]; });
  }
};
}  // namespace detail

// ラベル情報
class _EXPORT_IMGPROC_ LabelInfo : boost::addable<LabelInfo> {
  cv::Point m_SumPos;  // 座標の合計計算
  int m_SumPixVal;     // 座標値の合計
  cv::Point m_MinPos;  // 最小座標
  cv::Point m_MaxPos;  // 最大座標

 public:
  static const bool ENABLE_PARALLEL;      // 並列化の許可
  static const bool ENABLE_FINAL_UPDATE;  // 最終アップデート有効
  int LabelId;                            // ラベルID
  int NumOfPix;                           // ピクセル数
  double AveVal;                          // 平均画素値
  cv::Rect Area;                          // バウンダリーボックス情報
  cv::Point2d Centroid;                   // 重心

  // コンストラクタ
  explicit LabelInfo(int labelId);

  // 座標値の追加
  void Add(const cv::Point& pos, int val);

  // ラベル情報の計算
  void operator()();

  // +=演算子
  const LabelInfo& operator+=(const LabelInfo& rhs);

  // デバッグ画像作成
  template <typename BufType>
  static void CreateDbgImg(cv::Mat& dbg, const cv::Mat& labelingImg,
                           const std::vector<LabelInfo>& labelInfos, bool flag = false) {
    using namespace std;
    using namespace cv;
    using namespace tbb;
    using namespace commonutility;

    if (labelInfos.size()) {
      if (dbg.size() != labelingImg.size()) dbg = Mat(labelingImg.size(), CV_8UC3);

      tbb_for<int>(0, labelInfos.size(), [&](int i) {
        const LabelInfo& itm = labelInfos[i];
        tbb_for(itm.Area, [&](const blocked_range2d<int>& range) {
          for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
            int idxY = itm.Area.y + y;
            for (int x = std::begin(range.cols()); x < std::end(range.cols()); ++x) {
              int idxX = itm.Area.x + x;
              if (labelingImg.at<BufType>(idxY, idxX) == itm.LabelId)
                dbg.at<Vec3b>(idxY, idxX) = ColorTable[i];
            }
          }
        });
      });

      if (flag) {
        tbb_for_each(labelInfos, [&](const LabelInfo& itm) {
          circle(dbg, static_cast<cv::Point>(itm.Centroid), 10, CV_RGB(0, 0, 0));
          rectangle(dbg, itm.Area, CV_RGB(0, 0, 0));
        });
      }
    }
  }
};

// ラベル情報
struct _EXPORT_IMGPROC_ LabelInfoSimple {
  static const bool ENABLE_PARALLEL;      // 並列化の許可
  static const bool ENABLE_FINAL_UPDATE;  // 最終アップデート有効
  int LabelId;                            // ラベルID
  std::atomic<int> NumOfPix;              // ピクセル数

  // コンストラクタ
  explicit LabelInfoSimple(int labelId = 0);

  // コピーコンストラクタ
  LabelInfoSimple(const LabelInfoSimple& rhs);

  // 代入演算子
  LabelInfoSimple& operator=(const LabelInfoSimple& rhs);

  // 座標値の追加
  void Add(const cv::Point&, int);

  // ラベル情報の計算
  void operator()();

  // デバッグ画像作成
  template <typename BufType>
  static void CreateDbgImg(cv::Mat& dbg, const cv::Mat& labelingImg,
                           const std::vector<LabelInfoSimple>& labelInfos) {
    using namespace std;
    using namespace cv;
    using namespace tbb;
    using namespace commonutility;

    if (labelInfos.size()) {
      if (dbg.size() != labelingImg.size()) dbg = Mat(labelingImg.size(), CV_8UC3);

      serial_for(blocked_range3d<int>(0, static_cast<int>(labelInfos.size()), 0, labelingImg.rows,
                                      0, labelingImg.cols),
                 [&](const blocked_range3d<int>& range) {
                   for (int i = std::begin(range.pages()); i < std::end(range.pages()); ++i) {
                     const LabelInfoSimple& itm = labelInfos[i];
                     const Vec3b color = ColorTable[i];
                     for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
                       int idxY = y * labelingImg.cols + std::begin(range.cols());
                       uchar* ptrDbg = dbg.ptr<uchar>() + idxY * 3;
                       const BufType* ptrLabelingImg = labelingImg.ptr<BufType>() + idxY;
                       for (int x = std::begin(range.cols()); x < std::end(range.cols());
                            ++x, ++ptrLabelingImg) {
                         if (*ptrLabelingImg == itm.LabelId)
                           for (int c = 0; c < 3; ++c) *ptrDbg++ = color[c];
                         else
                           ptrDbg += 3;
                       }
                     }
                   }
                 });
    }
  }
};

// 高速ラベリングクラス(4近傍最適化)
template <typename InputType, typename BufType = ushort>
struct Labeling {
  using ImgType = BufType;
  static constexpr size_t MAX_NUM_LABEL = 1 << (sizeof(BufType) * 8);  // 最大ラベル数
  static constexpr size_t MAX_LABEL_INDEX = MAX_NUM_LABEL - 1;         // 最大ラベルインデックス

 private:
  detail::LabelManager<MAX_NUM_LABEL> m_NumOfLabel;  // ラベル管理
  std::vector<BufType> m_LabelingImg;                // ラベル画像
  cv::Size m_LabelingImgSize;                        // ラベル画像サイズ

  // ラベルのアップデート(比較画素1つ版)
  template <typename F>
  void UpdateLabel1(const InputType& valPrv, const InputType& val, const BufType& labelPrv,
                    BufType& label, F& func) {
    if (func(val, valPrv))  // 前の画素と近い
      label = labelPrv;     // ラベル番号のコピー
    else                    // 前の画素と違う場合
    {
      label = static_cast<BufType>(++m_NumOfLabel);  // 新規ラベル番号設定

      // ラベル数オーバーチェック
      Assert(static_cast<size_t>(m_NumOfLabel) < MAX_LABEL_INDEX, "over num of label");
    }
  }

  // ラベルのアップデート(比較画素2つ版)
  template <typename F>
  void UpdateLabel2(const InputType& valPrvX, const InputType& valPrvY, const InputType& val,
                    const BufType& labelPrvX, const BufType& labelPrvY, BufType& label, F& func) {
    const bool flagX = func(val, valPrvX);
    const bool flagY = func(val, valPrvY);
    if (flagY) {
      label = labelPrvY;  // ラベル番号のコピー

      if (flagX && labelPrvX != labelPrvY) m_NumOfLabel(labelPrvX, labelPrvY);
    } else {
      if (flagX)
        label = labelPrvX;  // ラベル番号のコピー
      else {
        label = static_cast<BufType>(++m_NumOfLabel);  // 新規ラベル番号設定
        Assert(static_cast<size_t>(m_NumOfLabel) < MAX_LABEL_INDEX,
               "over num of label");  // ラベル数オーバーチェック
      }
    }
  }

 public:
  // コンストラクタ
  Labeling() : m_NumOfLabel(), m_LabelingImg(), m_LabelingImgSize() {}

  // コンストラクタ
  explicit Labeling(const cv::Size& size)
      : m_NumOfLabel(), m_LabelingImg(size.area()), m_LabelingImgSize(size) {}

  // 実行
  template <typename F>
  void operator()(const cv::Mat& img, F& func) {
    using namespace tbb;

    // サイズ不一致の例外処理
    Assert(m_LabelingImgSize == img.size(), "bad image size");

    // ラベリング画像作成
    cv::Mat labelingImg(m_LabelingImgSize, cv::Type<BufType>(), &m_LabelingImg[0]);

    m_NumOfLabel = 0;  // ラベルカウンタ初期化

    const int& srcCols = img.cols;               // 入力画像のステップ量
    const int& labelingCols = labelingImg.cols;  // ラベル画像のステップ量

    // 1行目(左の画素だけを見る)
    {
      const InputType* ptrPrv = img.ptr<InputType>();     // 入力画像の一つ前
      const InputType* ptr = ptrPrv + 1;                  // 入力画像の現在地
      const InputType* ptrEnd = ptrPrv + srcCols;         // 一行目のラスト
      BufType* ptrLabelPrv = labelingImg.ptr<BufType>();  // ラベル画像の一つ前
      BufType* ptrLabel = ptrLabelPrv + 1;                // ラベル画像の現在値

      // 左だけチェック
      for (*ptrLabelPrv = 0; ptr < ptrEnd; ++ptrPrv, ++ptr, ++ptrLabelPrv, ++ptrLabel) {
        UpdateLabel1(*ptrPrv, *ptr, *ptrLabelPrv, *ptrLabel, func);
      }
    }

    // 2行目以降
    for (int y = 1; y < img.rows; ++y) {
      const InputType* ptrPrvX = img.ptr<InputType>(y);     // 入力画像の一つ前
      const InputType* ptrPrvY = ptrPrvX - srcCols;         // 入力画像の一つ上
      BufType* ptrLabelPrvX = labelingImg.ptr<BufType>(y);  // ラベル画像の一つ前
      BufType* ptrLabelPrvY = ptrLabelPrvX - labelingCols;  // ラベル画像の一つ上

      // 1列目(上だけチェック)
      UpdateLabel1(*ptrPrvY++, *ptrPrvX, *ptrLabelPrvY++, *ptrLabelPrvX, func);

      // 2列目以降
      const InputType* ptr = ptrPrvX + 1;           // 入力画像の現在地
      const InputType* ptrEnd = ptrPrvX + srcCols;  // 位置行目のラスト
      BufType* ptrLabel = ptrLabelPrvX + 1;         // ラベル画像の現在値

      // 左->上の順でチェック
      for (; ptr < ptrEnd;
           ++ptrPrvX, ++ptrPrvY, ++ptr, ++ptrLabelPrvX, ++ptrLabelPrvY, ++ptrLabel) {
        UpdateLabel2(*ptrPrvX, *ptrPrvY, *ptr, *ptrLabelPrvX, *ptrLabelPrvY, *ptrLabel, func);
      }  // ラベルのアップデート
    }

    // ラベル統合
    m_NumOfLabel.IntegratedLabel();

#ifdef USE_TBB
    // ラベル番号置き換え
    tbb_for(blocked_range<BufType*>(
                labelingImg.ptr<BufType>(),
                labelingImg.ptr<BufType>(labelingImg.rows - 1, labelingImg.cols - 1) + 1),
            [this](const blocked_range<BufType*>& range) {
              for (BufType* ptr = std::begin(range); ptr < std::end(range); ++ptr) {
                const int& lutVal = m_NumOfLabel[*ptr];
                if (lutVal != *ptr) *ptr = static_cast<BufType>(lutVal);
              }
            });
#endif
  }

  // 実行
  void CompEqual(const cv::Mat& img) {
    std::equal_to<InputType> func;
    (*this)(img, func);
  }

  // 実行
  void CompAbsDif(const cv::Mat& img, int difVal = 10) {
    labeling::AbsDif func(difVal);
    (*this)(img, func);
  }

  // ラベル情報作成
  template <typename Info>
  void CreateLabelInfo(const cv::Mat& img, std::vector<Info>& labelInfos) const {
    using namespace std;
    using namespace cv;
    using namespace tbb;

    const Mat labelingImg(m_LabelingImgSize, cv::Type<BufType>(),
                          const_cast<BufType*>(&m_LabelingImg[0]));

    labelInfos.clear();

    for (int i = 0; i < static_cast<int>(m_NumOfLabel); ++i) labelInfos.emplace_back(i);

    const blocked_range2d<int> setRange(0, labelingImg.rows, 0, labelingImg.cols);
    auto imgSearch = [&](const blocked_range2d<int>& range) {
      for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
        Point pos(std::begin(range.cols()), y);
        int& x = pos.x;
        const int idxY = y * labelingImg.cols + x;
        const BufType* ptr = labelingImg.ptr<BufType>() + idxY;
        const InputType* ptrSrc = img.ptr<InputType>() + idxY;
        for (; x < std::end(range.cols()); ++x, ++ptr, ++ptrSrc)
          labelInfos[*ptr].Add(pos, static_cast<int>(*ptrSrc));
      }
    };

    if (Info::ENABLE_PARALLEL)
      tbb_for(setRange, imgSearch);
    else
      serial_for(setRange, imgSearch);

    if (Info::ENABLE_FINAL_UPDATE) {
      tbb_for_each(labelInfos, [](Info& itm) { itm(); });
    }
  }

  // ラベル画像取得
  const cv::Mat GetLabelingImg() const {
    return {m_LabelingImgSize, cv::Type<BufType>(), const_cast<BufType*>(&m_LabelingImg[0])};
  }

  // ラベル数取得
  size_t GetNumOfLabel() const { return m_NumOfLabel; }

  // デバッグ画像作成
  void CreateDbgImg(cv::Mat& dbg) const {
    using namespace std;
    using namespace cv;
    using namespace tbb;
    using namespace commonutility;

    if (dbg.size() != m_LabelingImg.size()) dbg = Mat(m_LabelingImg.size(), CV_8UC3);

    tbb_for<int>(0, static_cast<int>(dbg.size().area()), [&, this](int i) {
      dbg.at<Vec3b>(i) = ColorTable[saturate_cast<int>(m_LabelingImg.at<Labeling::BufType>(i))];
    });
  }

  // デバッグ画像作成
  template <typename Info>
  void CreateDbgImg(cv::Mat& dbg, const std::vector<Info>& labelInfos) const {
    Info::template CreateDbgImg<BufType>(
        dbg, Mat(m_LabelingImgSize, cv::Type<BufType>(), const_cast<BufType*>(&m_LabelingImg[0])),
        labelInfos);
  }
};

// 高速マルチスレッドラベリングクラス(4近傍最適化)
template <typename CompFunc, typename InputType, typename BufType = ushort>
struct LabelingFast {
  using ImgType = BufType;
  static constexpr size_t MAX_NUM_LABEL = 1 << (sizeof(BufType) * 8);  // 最大ラベル数
  static constexpr size_t MAX_LABEL_INDEX = MAX_NUM_LABEL - 1;         // 最大ラベルインデックス

 private:
  using Type = LabelingFast<CompFunc, InputType, BufType>;

  detail::LabelManager<MAX_NUM_LABEL> m_NumOfLabel;  // ラベル管理
  cv::Mat m_LabelingImg;                             // ラベル画像

  cv::Size m_ImgSize;     // 画像サイズ
  CompFunc m_Func;        // 比較関数
  cv::Size m_NumOfBlock;  // ブロックサイズ
  cv::Mat m_SrcImg;       // 入力画像

  tbb::flow::graph m_Glaph;                                               // グラフ
  std::vector<tbb::flow::continue_node<tbb::flow::continue_msg>> m_Node;  // ノード

  // ラベルのアップデート(比較画素1つ版)
  void UpdateLabel1(const InputType& valPrv, const InputType& val, const BufType& labelPrv,
                    BufType& label) {
    if (m_Func(val, valPrv))  // 前の画素と近い
      label = labelPrv;       // ラベル番号のコピー
    else                      // 前の画素と違う場合
    {
      label = static_cast<BufType>(++m_NumOfLabel);  // 新規ラベル番号設定

      // ラベル数オーバーチェック
      Assert(static_cast<size_t>(m_NumOfLabel) < MAX_LABEL_INDEX, "over num of label");
    }
  }

  // ラベルのアップデート(比較画素2つ版)
  void UpdateLabel2(const InputType& valPrvX, const InputType& valPrvY, const InputType& val,
                    const BufType& labelPrvX, const BufType& labelPrvY, BufType& label) {
    const bool flagX = m_Func(val, valPrvX);
    const bool flagY = m_Func(val, valPrvY);
    if (flagY) {
      label = labelPrvY;  // ラベル番号のコピー

      if (flagX && labelPrvX != labelPrvY) m_NumOfLabel(labelPrvX, labelPrvY);
    } else {
      if (flagX)
        label = labelPrvX;  // ラベル番号のコピー
      else {
        label = static_cast<BufType>(++m_NumOfLabel);  // 新規ラベル番号設定
        Assert(static_cast<size_t>(m_NumOfLabel) < MAX_LABEL_INDEX,
               "over num of label");  // ラベル数オーバーチェック
      }
    }
  }

  // 初期化
  void Init() {
    using namespace std;
    using namespace cv;
    using namespace tbb::flow;

    // 初期設定
    const Size blockSize(
        cvCeil(static_cast<double>(m_ImgSize.width) / m_NumOfBlock.width),
        cvCeil(static_cast<double>(m_ImgSize.height) / m_NumOfBlock.height));  // 分割数セット

    // ラベル画像パラメータのエイリアス作成
    const int& labelingCols = m_LabelingImg.cols;
    const int& labelingRows = m_LabelingImg.rows;

    // ノードの処理を設定
    for (int j = 0; j < m_NumOfBlock.height; ++j) {
      for (int i = 0; i < m_NumOfBlock.width; ++i) {
        m_Node.emplace_back(m_Glaph, [blockSize, labelingCols, labelingRows, j, i,
                                      this](const continue_msg&) {
          // 入力画像パラメータのエイリアス作成
          const int& srcCols = m_SrcImg.cols;  // 入力画像のステップ量

          // xの範囲
          const int posX = i * blockSize.width;
          const int nextX = posX + blockSize.width;
          const Range rangeX(posX, min(nextX, labelingCols));
          const int xRangeSize = rangeX.size();

          // yの範囲
          const int posY = j * blockSize.height;
          const int nextY = posY + blockSize.height;
          const Range rangeY(posY, min(nextY, labelingRows));

          // ポインタ作成
          const InputType* ptrBaseY =
              m_SrcImg.ptr<InputType>(rangeY.start, rangeX.start);  // 入力画像のポインタ
          const InputType* ptrEndY =
              m_SrcImg.ptr<InputType>(rangeY.end - 1, rangeX.end - 1) + 1;  // 終了位置
          BufType* ptrLabelBaseY =
              m_LabelingImg.ptr<BufType>(rangeY.start, rangeX.start);  // ラベル画像のポインタ

          // 1行目(左の画素だけを見る)
          if (!rangeY.start) {
            const InputType* ptr = ptrBaseY;             // 入力画像のポインタ
            const InputType* ptrEnd = ptr + xRangeSize;  // 範囲の終了ポインタ
            BufType* ptrLabel = ptrLabelBaseY;           // ラベル画像のポインタ

            // 1列目(ラベル番号の初期化)
            if (!rangeX.start) {
              ++ptr;            // 入力画像のポインタを進める
              *ptrLabel++ = 0;  // ラベル番号初期化&ラベル画像のポインタを進める
            }

            // 2列目以降
            const InputType* ptrPrv = ptr - 1;          // 左の入力画素のポインタ
            const BufType* ptrLabelPrv = ptrLabel - 1;  // 左のラベル画像のポインタ
            for (; ptr < ptrEnd; ++ptrPrv, ++ptr, ++ptrLabelPrv, ++ptrLabel)
              this->UpdateLabel1(*ptrPrv, *ptr, *ptrLabelPrv, *ptrLabel);  // ラベルのアップデート

            // Yのスタート位置を進める
            ptrBaseY += srcCols;
            ptrLabelBaseY += labelingCols;
          }

          // 2行目以降
          for (; ptrBaseY < ptrEndY; ptrBaseY += srcCols, ptrLabelBaseY += labelingCols) {
            const InputType* ptr = ptrBaseY;                        // 入力画像のポインタ
            const InputType* ptrPrvY = ptr - srcCols;               // 上の入力画像のポインタ
            const InputType* ptrEnd = ptr + xRangeSize;             // 範囲の終了ポインタ
            BufType* ptrLabel = ptrLabelBaseY;                      // ラベル画像のポインタ
            const BufType* ptrLabelPrvY = ptrLabel - labelingCols;  // 上のラベル画像のポインタ

            // 1列目(上の画素だけを見る)
            if (!rangeX.start)
              this->UpdateLabel1(*ptrPrvY++, *ptr++, *ptrLabelPrvY++, *ptrLabel++);  // ラベルのアップデート

            // 2列目以降
            const InputType* ptrPrvX = ptr - 1;          // 左の入力画素のポインタ
            const BufType* ptrLabelPrvX = ptrLabel - 1;  // 左のラベル画像のポインタ
            for (; ptr < ptrEnd;
                 ++ptrPrvX, ++ptrPrvY, ++ptr, ++ptrLabelPrvX, ++ptrLabelPrvY, ++ptrLabel) {
              this->UpdateLabel2(*ptrPrvX, *ptrPrvY, *ptr, *ptrLabelPrvX, *ptrLabelPrvY, *ptrLabel);
            }  // ラベルのアップデート
          }
        });
      }
    }

    // ノード間を繋ぐエッジを作成
    for (int j = 0, idx = 0; j < m_NumOfBlock.height; ++j) {
      for (int i = 0; i < m_NumOfBlock.width; ++i, ++idx) {
        if (i + 1 < m_NumOfBlock.width) make_edge(m_Node[idx], m_Node[idx + 1]);

        if (j + 1 < m_NumOfBlock.height) make_edge(m_Node[idx], m_Node[idx + m_NumOfBlock.width]);
      }
    }
  }

 public:
  // コンストラクタ
  LabelingFast()
      : m_NumOfLabel(),
        m_LabelingImg(),
        m_ImgSize(),
        m_Func(),
        m_NumOfBlock(),
        m_SrcImg(),
        m_Glaph(),
        m_Node() {}

  // コンストラクタ
  LabelingFast(const cv::Size& size, const cv::Size& numOfBlock = cv::Size(20, 20))
      : m_NumOfLabel(),
        m_LabelingImg(cv::Mat::zeros(size, cv::Type<BufType>())),
        m_ImgSize(size),
        m_Func(),
        m_NumOfBlock(numOfBlock),
        m_SrcImg(),
        m_Glaph(),
        m_Node() {
    Init();
  }

  // コンストラクタ
  LabelingFast(const cv::Size& size, const CompFunc& func,
               const cv::Size& numOfBlock = cv::Size(20, 20))
      : m_NumOfLabel(),
        m_LabelingImg(cv::Mat::zeros(size, cv::Type<BufType>())),
        m_ImgSize(size),
        m_Func(func),
        m_NumOfBlock(numOfBlock),
        m_SrcImg(),
        m_Glaph(),
        m_Node() {
    Init();
  }

  // コピーコンストラクタ
  LabelingFast(const Type& rhs)
      : m_NumOfLabel(rhs.m_NumOfLabel),
        m_LabelingImg(rhs.m_LabelingImg.clone()),
        m_ImgSize(rhs.m_ImgSize),
        m_Func(rhs.m_Func),
        m_NumOfBlock(rhs.m_NumOfBlock),
        m_SrcImg(),
        m_Glaph(),
        m_Node() {
    Init();
  }

  // swap
  void swap(Type& rhs) {
    std::swap(m_NumOfLabel, rhs.m_NumOfLabel);
    std::swap(m_LabelingImg, rhs.m_LabelingImg);
    std::swap(m_ImgSize, rhs.m_ImgSize);
    std::swap(m_Func, rhs.m_Func);
    std::swap(m_NumOfBlock, rhs.m_NumOfBlock);

    m_Node.clear();
    Init();
  }

  // 代入演算子
  Type& operator=(const Type& rhs) {
    m_NumOfLabel = rhs.m_NumOfLabel;
    rhs.m_LabelingImg.copyTo(m_LabelingImg);
    m_ImgSize = rhs.m_ImgSize;
    m_Func = rhs.m_Func;
    m_NumOfBlock = rhs.m_NumOfBlock;

    m_Node.clear();
    Init();

    return *this;
  }

  // 実行
  void operator()(cv::Mat& img) {
    using namespace tbb;
    using namespace tbb::flow;

    // サイズ不一致の例外処理
    Assert(m_LabelingImg.size() == img.size(), "bad image size");

    m_NumOfLabel = 0;  // ラベルカウンタ初期化
    m_SrcImg = img;    // 入力画像のセット
    m_LabelingImg.at<BufType>() = 0;

    m_Node[0].try_put(continue_msg());  // 処理開始
    m_Glaph.wait_for_all();             // 処理の終了待ち

    // ラベル統合
    m_NumOfLabel.IntegratedLabel();

    // ラベル番号置き換え
    tbb_for(blocked_range<BufType*>(reinterpret_cast<BufType*>(m_LabelingImg.datastart),
                                    reinterpret_cast<BufType*>(m_LabelingImg.dataend)),
            [this](const blocked_range<BufType*>& range) {
              for (BufType* ptr = std::begin(range); ptr < std::end(range); ++ptr) {
                const int& lutVal = m_NumOfLabel[*ptr];
                if (lutVal != *ptr) *ptr = static_cast<BufType>(lutVal);
              }
            });
  }

  // ラベル情報作成
  template <typename Info>
  void CreateLabelInfo(const cv::Mat& img, std::vector<Info>& labelInfos) const {
    using namespace std;
    using namespace cv;
    using namespace tbb;

    labelInfos.clear();

    for (int i = 0; i < static_cast<int>(m_NumOfLabel); ++i) labelInfos.emplace_back(i);

    const blocked_range2d<int> setRange(0, m_LabelingImg.rows, 0, m_LabelingImg.cols);
    auto imgSearch = [&](const blocked_range2d<int>& range) {
      for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
        Point pos(std::begin(range.cols()), y);
        int& x = pos.x;
        const int idxY = y * m_LabelingImg.cols + x;
        const BufType* ptr = m_LabelingImg.ptr<BufType>() + idxY;
        const InputType* ptrSrc = img.ptr<InputType>() + idxY;
        for (; x < std::end(range.cols()); ++x, ++ptr, ++ptrSrc)
          labelInfos[*ptr].Add(pos, static_cast<int>(*ptrSrc));
      }
    };

    if (Info::ENABLE_PARALLEL)
      tbb_for(setRange, imgSearch);
    else
      serial_for(setRange, imgSearch);

    if (Info::ENABLE_FINAL_UPDATE) {
      tbb_for_each(labelInfos, [](Info& itm) { itm(); });
    }
  }

  // ラベル画像取得
  const cv::Mat& GetLabelingImg() const { return m_LabelingImg; }

  // ラベル数取得
  size_t GetNumOfLabel() const { return m_NumOfLabel; }

  // デバッグ画像作成
  void CreateDbgImg(cv::Mat& dbg) const {
    using namespace std;
    using namespace cv;
    using namespace tbb;
    using namespace commonutility;

    if (dbg.size() != m_LabelingImg.size()) dbg = Mat(m_LabelingImg.size(), CV_8UC3);

    tbb_for<int>(0, static_cast<int>(dbg.size().area()), [&, this](int i) {
      dbg.at<Vec3b>(i) = ColorTable[saturate_cast<int>(*(m_LabelingImg.ptr<BufType>(i)))];
    });
  }

  // デバッグ画像作成
  template <typename Info>
  void CreateDbgImg(cv::Mat& dbg, const std::vector<Info>& labelInfos) const {
    Info::template CreateDbgImg<BufType>(dbg, m_LabelingImg, labelInfos);
  }
};
}  // namespace imgproc

MSVC_WARNING_POP
