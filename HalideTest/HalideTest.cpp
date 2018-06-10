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
  const auto srcImg = cv::imread(R"(C:\Library\Source\opencv\samples\data\lena.jpg)");
  BOOST_TEST_MESSAGE("{:d}x{:d}"_format(srcImg.cols, srcImg.rows));

  {
    cv::Mat dstImg(srcImg.rows << 1, srcImg.cols << 1, srcImg.type());
    BOOST_TEST_MESSAGE("{:d}x{:d}"_format(dstImg.cols, dstImg.rows));

    const auto t = chr::high_resolution_clock::now();
    for (auto i = 0; i < 10; ++i) {
      const auto t_ = chr::high_resolution_clock::now();
      rawsample::ResizeWithRawAccess(srcImg, dstImg);
      BOOST_TEST_MESSAGE(
          chr::duration_cast<chr::milliseconds>(chr::high_resolution_clock::now() - t_).count());
    }
    BOOST_TEST_MESSAGE(
        chr::duration_cast<chr::milliseconds>(chr::high_resolution_clock::now() - t).count());

    cv::imshow("DstRaw", dstImg);
  }
  {
    cv::Mat dstImg(srcImg.rows << 1, srcImg.cols << 1, srcImg.type());
    BOOST_TEST_MESSAGE("{:d}x{:d}"_format(dstImg.cols, dstImg.rows));

    const auto t = chr::high_resolution_clock::now();
    halidesample::ResizeWithHalide(srcImg, dstImg);
    BOOST_TEST_MESSAGE(
        chr::duration_cast<chr::milliseconds>(chr::high_resolution_clock::now() - t).count());

    cv::imshow("DstHalide", dstImg);
  }
  cv::waitKey();
}

BOOST_AUTO_TEST_SUITE_END()
