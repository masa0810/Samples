#include "ResizeSample.h"

#pragma warning(push, 0)

#include <chrono>

#include <boost/test/unit_test.hpp>

#include <fmt/format.h>

#include <Halide.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#pragma warning(pop)

using namespace fmt::literals;

namespace chr = std::chrono;

BOOST_AUTO_TEST_SUITE(Common)

BOOST_AUTO_TEST_CASE(Init) {
  const auto srcImg =
      cv::imread(R"(C:\Library\Current\Source\opencv\samples\data\lena.jpg)");
  BOOST_TEST_MESSAGE("{:d}x{:d}"_format(srcImg.cols, srcImg.rows));

  cv::Mat dstImg(srcImg.rows << 1, srcImg.cols << 1, srcImg.type());
  BOOST_TEST_MESSAGE("{:d}x{:d}"_format(dstImg.cols, dstImg.rows));

  rawsample::ResizeWithRawAccess(srcImg, dstImg);

  cv::imshow("Dst", dstImg);
  cv::waitKey();
}

BOOST_AUTO_TEST_SUITE_END()
