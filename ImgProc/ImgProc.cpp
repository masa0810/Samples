#include "DistanceTransform.h"
#include "ImageCollection.h"

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <cmath>
#include <cstdint>

#include <benchmark/benchmark.h>

#include <opencv2/imgcodecs.hpp>

// 警告抑制解除
MSVC_WARNING_POP

namespace gbm = benchmark;
namespace img = imgproc;
namespace com = commonutility;

class Fixture : public gbm::Fixture {
 protected:
  void SetUp(const gbm::State& state) {
    tmp = cv::imread(R"(D:\Users\masah\Pictures\shape.pgm)", cv::IMREAD_GRAYSCALE);
    tmp.copyTo(srcMat);
    dstMat.create(tmp.size(), cv::Type<float>());

    srcBuf.Init(tmp.size());
    dstBuf.Init(tmp.size());

    auto src = srcBuf.Mat();
    tmp.copyTo(src);

    distTrans.Init(tmp.size());
  }

  void TearDown(const gbm::State& state) {
    tmp.release();
    srcMat.release();
    dstMat.release();
    srcBuf.Clear();
    dstBuf.Clear();
  }

  cv::Mat tmp = {};

  cv::Mat srcMat = {};
  cv::Mat dstMat = {};

  com::AlignedImageBuffer_<std::uint8_t, 3> srcBuf = {};
  com::AlignedImageBuffer_<float, 1> dstBuf = {};

  using DistanceTransform = img::DistanceTransform_<>;
  DistanceTransform distTrans;
};

BENCHMARK_F(Fixture, BM_DistTransTest)(gbm::State& st) {
  const auto& src = srcMat;
  auto& dst = dstMat;
  for (auto _ : st) {
    distTrans.CalcCompCast<std::uint8_t, float>(
        src, dst, [](const auto val) { return val != 0; },
        [](const auto val) { return cv::saturate_cast<float>(std::sqrt(val)); });
  }
}

BENCHMARK_F(Fixture, BM_DistTransTest2)(gbm::State& st) {
  const auto src = srcBuf.Mat();
  auto dst = dstBuf.Mat();
  for (auto _ : st) {
    distTrans.CalcCompCast<std::uint8_t, float>(
        src, dst, [](const auto val) { return val != 0; },
        [](const auto val) { return cv::saturate_cast<float>(std::sqrt(val)); });
  }
}
