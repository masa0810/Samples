#include "Labeling.h"

namespace imgproc {
namespace labeling {
// 絶対値比較
AbsDif::AbsDif(int difVal) : m_DifVal(difVal) {}
}  // namespace labeling

#pragma region ラベル情報

const bool LabelInfo::ENABLE_PARALLEL = false;     // 並列化の許可
const bool LabelInfo::ENABLE_FINAL_UPDATE = true;  // 最終アップデート有効

// コンストラクタ
LabelInfo::LabelInfo(int labelId = 0)
    : m_SumPos()  // 座標の合計計算
      ,
      m_SumPixVal(0)  // 座標値の合計
      ,
      m_MinPos(std::numeric_limits<int>::max(), std::numeric_limits<int>::max())  // 最小座標
      ,
      m_MaxPos(std::numeric_limits<int>::lowest(), std::numeric_limits<int>::lowest())  // 最大座標
      ,
      LabelId(labelId)  // ラベルID
      ,
      NumOfPix(0)  // ピクセル数
      ,
      AveVal(0.0)  // 平均画素値
      ,
      Area()  // バウンダリーボックス情報
      ,
      Centroid()  // 重心
{}

// 座標値の追加
void LabelInfo::Add(const cv::Point& pos, int val) {
  m_SumPos += pos;
  m_SumPixVal += val;

  m_MinPos.x = std::min(m_MinPos.x, pos.x);
  m_MinPos.y = std::min(m_MinPos.y, pos.y);
  m_MaxPos.x = std::max(m_MaxPos.x, pos.x);
  m_MaxPos.y = std::max(m_MaxPos.y, pos.y);

  ++NumOfPix;
}

// ラベル情報の計算
void LabelInfo::operator()() {
  AveVal = static_cast<double>(m_SumPixVal) / NumOfPix;
  Area = cv::Rect(m_MinPos, m_MaxPos);
  Centroid.x = static_cast<double>(m_SumPos.x) / NumOfPix;
  Centroid.y = static_cast<double>(m_SumPos.y) / NumOfPix;
}

// +=演算子
const LabelInfo& LabelInfo::operator+=(const LabelInfo& rhs) {
  if (LabelId == rhs.LabelId) {
    m_SumPos += rhs.m_SumPos;
    m_SumPixVal += rhs.m_SumPixVal;

    m_MinPos.x = std::min(m_MinPos.x, rhs.m_MinPos.x);
    m_MinPos.y = std::min(m_MinPos.y, rhs.m_MinPos.y);
    m_MaxPos.x = std::max(m_MaxPos.x, rhs.m_MaxPos.x);
    m_MaxPos.y = std::max(m_MaxPos.y, rhs.m_MaxPos.y);

    NumOfPix += rhs.NumOfPix;
  }

  return *this;
}

#pragma endregion

#pragma region ラベル情報(シンプル)

const bool LabelInfoSimple::ENABLE_PARALLEL = true;       // 並列化の許可
const bool LabelInfoSimple::ENABLE_FINAL_UPDATE = false;  // 最終アップデート有効

// コンストラクタ
LabelInfoSimple::LabelInfoSimple(int labelId)
    : LabelId(labelId)  // ラベルID
      ,
      NumOfPix()  // ピクセル数
{
  NumOfPix = 0;
}

// コピーコンストラクタ
LabelInfoSimple::LabelInfoSimple(const LabelInfoSimple& rhs)
    : LabelId(rhs.LabelId)  // ラベルID
      ,
      NumOfPix()  // ピクセル数
{
  NumOfPix = static_cast<int>(rhs.NumOfPix);
}

// 代入演算子
LabelInfoSimple& LabelInfoSimple::operator=(const LabelInfoSimple& rhs) {
  LabelId = rhs.LabelId;
  NumOfPix = static_cast<int>(rhs.NumOfPix);
  return *this;
}

// 座標値の追加
void LabelInfoSimple::Add(const cv::Point&, int) { ++NumOfPix; }

// ラベル情報の計算
void LabelInfoSimple::operator()() {}

#pragma endregion
}  // namespace imgproc
