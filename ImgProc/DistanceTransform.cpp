#include "DistanceTransform.h"

#ifdef BENCHMARK_USE_MAIN

#include "ImageCollection.h"

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <cmath>
#include <cstdint>

#include <string>

#include <benchmark/benchmark.h>

#include <fmt/format.h>

#include <gtest/gtest.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// 警告抑制解除
MSVC_WARNING_POP

using namespace fmt::literals;
using namespace std::literals;

namespace gbm = benchmark;
namespace img = imgproc;
namespace com = commonutility;

static void DistTransTestOld(gbm::State& st, const std::string& fileName) {
  using DistanceTransform = img::old::DistanceTransform_<>;
  const auto src =
      cv::imread(R"(D:\Users\masah\Pictures\{:s}.jpg)"_format(fileName), cv::IMREAD_GRAYSCALE);
  cv::Mat dst(src.rows, src.cols, cv::Type<float>());
  DistanceTransform distTrans;
  distTrans.Init(src.size());
  for (auto _ : st) {
    distTrans.CalcCompCast<std::uint8_t, float>(
        src, dst, [](const auto val) { return val != 0; },
        [](const auto val) { return cv::saturate_cast<float>(std::sqrt(val)); });
  }
}
static void DistTransTest_1(gbm::State& st, const std::string& fileName) {
  using DistanceTransform = img::DistanceTransform_<>;
  const auto src =
      cv::imread(R"(D:\Users\masah\Pictures\{:s}.jpg)"_format(fileName), cv::IMREAD_GRAYSCALE);
  cv::Mat dst(src.rows, src.cols, cv::Type<float>());
  DistanceTransform distTrans;
  distTrans.Init(src.size());
  for (auto _ : st) {
    distTrans.Calc<std::uint8_t, float>(src, dst);
  }
}
static void DistTransTest_2(gbm::State& st, const std::string& fileName) {
  using DistanceTransform = img::DistanceTransform_<>;
  const auto tmp =
      cv::imread(R"(D:\Users\masah\Pictures\{:s}.jpg)"_format(fileName), cv::IMREAD_GRAYSCALE);

  com::AlignedImageBuffer_<std::uint8_t> srcBuf(tmp.size());
  com::AlignedImageBuffer_<float> dstBuf(tmp.size());
  DistanceTransform distTrans;
  distTrans.Init(tmp.size());

  auto src = srcBuf.Mat();
  auto dst = dstBuf.Mat();

  tmp.copyTo(src);
  for (auto _ : st) {
    distTrans.Calc<std::uint8_t, float>(src, dst);
  }
}
static void DistTransTest_OpenCV(gbm::State& st, const std::string& fileName) {
  const auto src =
      cv::imread(R"(D:\Users\masah\Pictures\{:s}.jpg)"_format(fileName), cv::IMREAD_GRAYSCALE);
  cv::Mat dst(src.rows, src.cols, cv::Type<float>());

  for (auto _ : st) {
    cv::distanceTransform(src, dst, cv::DistanceTypes::DIST_L2,
                          cv::DistanceTransformMasks::DIST_MASK_PRECISE);
  }
}
static void DistTransWidthIndexTest_1(gbm::State& st, const std::string& fileName) {
  using DistanceTransform = img::DistanceTransformWithIndex_<>;
  const auto src =
      cv::imread(R"(D:\Users\masah\Pictures\{:s}.jpg)"_format(fileName), cv::IMREAD_GRAYSCALE);
  cv::Mat dst(src.rows, src.cols, cv::Type<float>());
  DistanceTransform distTrans;
  distTrans.Init(src.size());
  for (auto _ : st) {
    distTrans.Calc<std::uint8_t, float>(src, dst);
  }
}
static void DistTransWidthIndexTest_2(gbm::State& st, const std::string& fileName) {
  using DistanceTransform = img::DistanceTransformWithIndex_<>;
  const auto tmp =
      cv::imread(R"(D:\Users\masah\Pictures\{:s}.jpg)"_format(fileName), cv::IMREAD_GRAYSCALE);

  com::AlignedImageBuffer_<std::uint8_t> srcBuf(tmp.size());
  com::AlignedImageBuffer_<float> dstBuf(tmp.size());
  DistanceTransform distTrans;
  distTrans.Init(tmp.size());

  auto src = srcBuf.Mat();
  auto dst = dstBuf.Mat();

  tmp.copyTo(src);
  for (auto _ : st) {
    distTrans.Calc<std::uint8_t, float>(src, dst);
  }
}
BENCHMARK_CAPTURE(DistTransTestOld, DistTransTest1, "horse"s);
BENCHMARK_CAPTURE(DistTransTest_1, DistTransTest1, "horse"s);
BENCHMARK_CAPTURE(DistTransTest_2, DistTransTest1, "horse"s);
BENCHMARK_CAPTURE(DistTransTest_OpenCV, DistTransTestOpenCv, "horse"s);
BENCHMARK_CAPTURE(DistTransWidthIndexTest_1, DistTransWidthIndexTest1, "horse"s);
BENCHMARK_CAPTURE(DistTransWidthIndexTest_2, DistTransWidthIndexTest1, "horse"s);
BENCHMARK_CAPTURE(DistTransTestOld, DistTransTestOldW, "horse_wide"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_1, DistTransTest1W, "horse_wide"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_2, DistTransTest2W, "horse_wide"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_OpenCV, DistTransTestOpenCvW, "horse_wide"s)
    ->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransWidthIndexTest_1, DistTransWidthIndexTest1W, "horse_wide"s)
    ->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransWidthIndexTest_2, DistTransWidthIndexTest2W, "horse_wide"s)
    ->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTestOld, DistTransTestOldH, "horse_height"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_1, DistTransTest1H, "horse_height"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_2, DistTransTest2H, "horse_height"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_OpenCV, DistTransTestOpenCvH, "horse_height"s)
    ->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransWidthIndexTest_1, DistTransWidthIndexTest1H, "horse_height"s)
    ->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransWidthIndexTest_2, DistTransWidthIndexTest2H, "horse_height"s)
    ->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTestOld, DistTransTestOldL, "horse_large"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_1, DistTransTest1L, "horse_large"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_2, DistTransTest2L, "horse_large"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_OpenCV, DistTransTestOpenCvL, "horse_large"s)
    ->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransWidthIndexTest_1, DistTransWidthIndexTest1L, "horse_large"s)
    ->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransWidthIndexTest_2, DistTransWidthIndexTest2L, "horse_large"s)
    ->Unit(gbm::kMicrosecond);

#endif

#ifdef GTEST_USE_MAIN

#include "OpenCvConfig.h"

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <cstdint>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

// 警告抑制解除
MSVC_WARNING_POP

namespace test = testing;
namespace img = imgproc;

template <class T>
class DistansTransTest : public test::Test {
 protected:
  using FloatType = T;
  using OldType = img::old::DistanceTransform_<FloatType>;
  using NewType = img::DistanceTransform_<FloatType>;
  OldType oldVer;
  NewType newVer;

  DistansTransTest() = default;
  virtual ~DistansTransTest() = default;
};

using Types = test::Types<float, double>;

TYPED_TEST_CASE(DistansTransTest, Types);

TYPED_TEST(DistansTransTest, Compatibility) {
  const auto src = cv::imread(R"(D:\Users\masah\Pictures\horse.jpg)", cv::IMREAD_GRAYSCALE);
  this->oldVer.Init(src.size());
  this->newVer.Init(src.size());

  cv::Mat dstOld(src.rows, src.cols, cv::Type<FloatType>());
  cv::Mat dstNew(src.rows, src.cols, cv::Type<FloatType>());

  this->oldVer.Calc<std::uint8_t, FloatType>(src, dstOld);
  this->newVer.Calc<std::uint8_t, FloatType>(src, dstNew);
  cv::Mat comp;
  cv::bitwise_xor(dstOld, dstNew, comp);
  ASSERT_EQ(cv::countNonZero(comp), 0);

  this->oldVer.CalcSq<std::uint8_t, FloatType>(src, dstOld);
  this->newVer.CalcSq<std::uint8_t, FloatType>(src, dstNew);

  cv::bitwise_xor(dstOld, dstNew, comp);
  ASSERT_EQ(cv::countNonZero(comp), 0);

  //const auto idxXold = this->oldVer.GetIndexX();
  //const auto idxYold = this->oldVer.GetIndexY();

  //const auto& imgBufNew = this->newVer.GetImageBuffer();
  //const auto idxXnew = imgBufNew.Mat<NewType::Buf::X>();
  //const auto idxYnew = imgBufNew.Mat<NewType::Buf::Y>();

  //cv::bitwise_xor(idxXold, idxXnew, comp);
  //ASSERT_EQ(cv::countNonZero(comp), 0);
  //cv::bitwise_xor(idxYold, idxYnew, comp);
  //ASSERT_EQ(cv::countNonZero(comp), 0);

  //cv::Mat pointsOld(src.rows, src.cols, cv::Type<int, 2>());
  //cv::Mat pointsNew(src.rows, src.cols, cv::Type<int, 2>());

  //this->oldVer.CalcPoint(pointsOld);
  //this->newVer.CalcPoint(pointsNew);

  //cv::bitwise_xor(pointsOld, pointsNew, comp);
  //std::vector<cv::Mat> splitMat(comp.channels());
  //cv::split(comp, splitMat);
  //ASSERT_EQ(cv::countNonZero(splitMat[0]), 0);
  //ASSERT_EQ(cv::countNonZero(splitMat[1]), 0);
}

#endif
