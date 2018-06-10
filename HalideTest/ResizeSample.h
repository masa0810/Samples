#pragma once

#pragma warning(push, 0)

#include <opencv2/core.hpp>

#pragma warning(pop)

namespace rawsample {

void ResizeWithRawAccess(const cv::Mat& srcImg, cv::Mat& dstImg);
}

#pragma warning(push, 0)

#include <Halide.h>

#pragma warning(pop)

namespace halidesample {

void ResizeWithHalide(const cv::Mat& srcMat, cv::Mat& dstMat);

//class Resize : public Halide::Generator<Resize> {
// public:
//  Input<Func> Input = {"input", Halide::UInt(8), 3};
//  Output<Func> Output = { "output", Halide::UInt(8), 3 };
//
//  void Init(const cv::Mat& input, const float ratio);
//};

}  // namespace halidesample
