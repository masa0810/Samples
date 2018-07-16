#include "DistanceTransform.h"
#include "ImageCollection.h"

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <cmath>
#include <cstdint>

#include <string>

#include <benchmark/benchmark.h>

#include <fmt/format.h>

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

static void DistTransTestOld_1(gbm::State& st, const std::string& fileName) {
  using DistanceTransform = img::old::DistanceTransform_<>;
  const auto src =
      cv::imread(R"(D:\Users\masah\Pictures\{:s}.pgm)"_format(fileName), cv::IMREAD_GRAYSCALE);
  cv::Mat dst(src.rows, src.cols, cv::Type<float>());
  DistanceTransform distTrans;
  distTrans.Init(src.size());
  for (auto _ : st) {
    distTrans.CalcCompCast<std::uint8_t, float>(
        src, dst, [](const auto val) { return val != 0; },
        [](const auto val) { return cv::saturate_cast<float>(std::sqrt(val)); });
  }
}
static void DistTransTestOld_2(gbm::State& st, const std::string& fileName) {
  using DistanceTransform = img::old::DistanceTransform_<>;
  const auto tmp =
      cv::imread(R"(D:\Users\masah\Pictures\{:s}.pgm)"_format(fileName), cv::IMREAD_GRAYSCALE);

  com::AlignedImageBuffer_<std::uint8_t, 3> srcBuf(tmp.size());
  com::AlignedImageBuffer_<float, 1> dstBuf(tmp.size());
  DistanceTransform distTrans;
  distTrans.Init(tmp.size());

  auto src = srcBuf.Mat();
  auto dst = dstBuf.Mat();

  tmp.copyTo(src);
  for (auto _ : st) {
    distTrans.CalcCompCast<std::uint8_t, float>(
        src, dst, [](const auto val) { return val != 0; },
        [](const auto val) { return cv::saturate_cast<float>(std::sqrt(val)); });
  }
}
static void DistTransTest_1(gbm::State& st, const std::string& fileName) {
  using DistanceTransform = img::DistanceTransform_<>;
  const auto src =
      cv::imread(R"(D:\Users\masah\Pictures\{:s}.pgm)"_format(fileName), cv::IMREAD_GRAYSCALE);
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
      cv::imread(R"(D:\Users\masah\Pictures\{:s}.pgm)"_format(fileName), cv::IMREAD_GRAYSCALE);

  com::AlignedImageBuffer_<std::uint8_t, 3> srcBuf(tmp.size());
  com::AlignedImageBuffer_<float, 1> dstBuf(tmp.size());
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
      cv::imread(R"(D:\Users\masah\Pictures\{:s}.pgm)"_format(fileName), cv::IMREAD_GRAYSCALE);
  cv::Mat dst(src.rows, src.cols, cv::Type<float>());

  for (auto _ : st) {
    cv::distanceTransform(src, dst, cv::DistanceTypes::DIST_L2,
                          cv::DistanceTransformMasks::DIST_MASK_PRECISE);
  }
}
BENCHMARK_CAPTURE(DistTransTestOld_1, DistTransTest1, "shape"s);
BENCHMARK_CAPTURE(DistTransTestOld_2, DistTransTest1, "shape"s);
BENCHMARK_CAPTURE(DistTransTest_1, DistTransTest1, "shape"s);
BENCHMARK_CAPTURE(DistTransTest_2, DistTransTest1, "shape"s);
BENCHMARK_CAPTURE(DistTransTest_OpenCV, DistTransTestOpenCv, "shape"s);
BENCHMARK_CAPTURE(DistTransTestOld_1, DistTransTestOld1W, "shape_wide"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTestOld_2, DistTransTestOld2W, "shape_wide"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_1, DistTransTest1W, "shape_wide"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_2, DistTransTest2W, "shape_wide"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_OpenCV, DistTransTestOpenCvW, "shape_wide"s)
    ->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTestOld_1, DistTransTestOld1H, "shape_height"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTestOld_2, DistTransTestOld2H, "shape_height"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_1, DistTransTest1H, "shape_height"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_2, DistTransTest2H, "shape_height"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_OpenCV, DistTransTestOpenCvH, "shape_height"s)
    ->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTestOld_1, DistTransTestOld1L, "shape_large"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTestOld_2, DistTransTestOld2L, "shape_large"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_1, DistTransTest1L, "shape_large"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_2, DistTransTest2L, "shape_large"s)->Unit(gbm::kMicrosecond);
BENCHMARK_CAPTURE(DistTransTest_OpenCV, DistTransTestOpenCvL, "shape_large"s)
    ->Unit(gbm::kMicrosecond);
