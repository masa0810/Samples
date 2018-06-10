#include "ResizeSample.h"

#pragma warning(push, 0)

#include <cstdint>

#pragma warning(pop)

namespace rawsample {

const std::uint8_t* getPixel(const std::uint8_t* data, const int row, const int col,
                             const std::size_t step, const std::size_t elem_size) {
  return data + (row * step + col * elem_size);
}

std::uint8_t interpolate(const float c0, const float c1, const float c2, const float c3,
                         const std::uint8_t* p0, const std::uint8_t* p1, const std::uint8_t* p2,
                         const std::uint8_t* p3, const int i) {
  return cv::saturate_cast<std::uint8_t>(c0 * p0[i] + c1 * p1[i] + c2 * p2[i] + c3 * p3[i]);
}

void ResizeWithRawAccess(const cv::Mat& srcImg, cv::Mat& dstImg) {
  // 入力画像の各種値・ポインタを取り出す。
  const auto srcCols = srcImg.cols;
  const auto srcRows = srcImg.rows;
  const auto srcStep = static_cast<int>(srcImg.step);
  const auto srcElemSize = static_cast<int>(srcImg.elemSize());
  const auto* src_data = srcImg.data;

  // 出力画像の各種値・ポインタを取り出す。
  const auto dstCols = dstImg.cols;
  const auto dstRows = dstImg.rows;
  const auto dstStep = static_cast<int>(dstImg.step);
  const auto dstElemSize = static_cast<int>(dstImg.elemSize());
  auto* dstData = dstImg.data;

  // 拡大縮小率の逆数を計算する。
  const auto sc = static_cast<float>(srcCols) / dstCols;
  const auto sr = static_cast<float>(srcRows) / dstRows;

  // 拡大縮小後の画素を左上から順に走査する。
  // y軸方向の走査
  for (auto j = 0, jd = 0; j < dstRows; ++j, jd += dstStep) {
    // 元画像における位置yを算出する。
    const auto fj = j * sr;
    const auto cj0 = static_cast<int>(fj);  // 端数を切り捨てる。
    const auto dj = fj - cj0;

    // +1した値が画像内に収まるように調整する。
    const auto cj1 = cj0 + 1 >= srcRows ? cj0 : cj0 + 1;

    // 拡大縮小後画像へのポインタの位置を更新する。
    auto* ptrDst = dstData + jd;

    // x軸方向の走査
    for (auto i = 0, id = 0; i < dstCols; ++i, id += dstElemSize) {
      // 元画像における位置xを算出する。
      const auto fi = i * sc;
      const auto ci0 = static_cast<int>(fi);
      const auto di = fi - ci0;  // dx

      const auto ci1 = ci0 + 1 >= srcCols ? ci0 : ci0 + 1;

      // 面積を計算する。
      const auto c0 = (1.0f - dj) * (1.0f - di);
      const auto c1 = (1.0f - dj) * di;
      const auto c2 = dj * (1.0f - di);
      const auto c3 = dj * di;

      // 周辺画素を取り出す。
      const auto* srcPix0 = getPixel(src_data, cj0, ci0, srcStep, srcElemSize);
      const auto* srcPix1 = getPixel(src_data, cj0, ci1, srcStep, srcElemSize);
      const auto* srcPix2 = getPixel(src_data, cj1, ci0, srcStep, srcElemSize);
      const auto* srcPix3 = getPixel(src_data, cj1, ci1, srcStep, srcElemSize);

      // ポインタ位置を更新する。
      auto* dstPix = ptrDst + id;

      // RGB値を計算する。
      dstPix[0] = interpolate(c0, c1, c2, c3, srcPix0, srcPix1, srcPix2, srcPix3, 0);
      dstPix[1] = interpolate(c0, c1, c2, c3, srcPix0, srcPix1, srcPix2, srcPix3, 1);
      dstPix[2] = interpolate(c0, c1, c2, c3, srcPix0, srcPix1, srcPix2, srcPix3, 2);
    }
  }
}

}  // namespace rawsample

#pragma warning(push, 0)

#include <boost/test/unit_test.hpp>

#include <fmt/format.h>

#include <Halide.h>

#pragma warning(pop)

using namespace Halide;
namespace chr = std::chrono;

namespace halidesample {

void ResizeWithHalide(const cv::Mat& srcMat, cv::Mat& dstMat) {
  // 入力画像を読み込む。
  auto input = Halide::Buffer<std::uint8_t>::make_interleaved(
      const_cast<std::uint8_t*>(srcMat.ptr<std::uint8_t>()), srcMat.cols, srcMat.rows,
      srcMat.channels(), "input");

  // 画像内に収まるか否かの判定をなくすため、周辺画素を拡張する。
  Halide::Func srcImg("src_image");
  srcImg = Halide::BoundaryConditions::repeat_edge(input);

  // アルゴリズムを記述する。
  const int srcCols = input.width();
  const int srcRows = input.height();

  const float sc = static_cast<float>(srcCols) / dstMat.cols;
  const float sr = static_cast<float>(srcRows) / dstMat.rows;

  Halide::Var i{}, j{}, c{};

  // 元画像の位置を逆算する。
  auto fj = j * sr;

  // 端数を捨てる。
  auto cj0 = Halide::cast<int>(fj);

  // +1する。
  auto cj1 = cj0 + 1;

  // 距離を計算する。
  auto dj = fj - cj0;

  // 同じことを水平方向についても繰り返す。
  auto fi = i * sc;
  auto ci0 = Halide::cast<int>(fi);
  auto ci1 = ci0 + 1;
  auto di = fi - ci0;

  // 面積を計算する。
  const auto c0 = (1.0f - dj) * (1.0f - di);
  const auto c1 = (1.0f - dj) * di;
  const auto c2 = dj * (1.0f - di);
  const auto c3 = dj * di;

  // 周辺画素を取り出す。
  const auto& srcPix0 = srcImg(ci0, cj0, c);
  const auto& srcPix1 = srcImg(ci1, cj0, c);
  const auto& srcPix2 = srcImg(ci0, cj1, c);
  const auto& srcPix3 = srcImg(ci1, cj1, c);

  // 画素値を計算する。
  Halide::Func dstImg{};
  dstImg(i, j, c) = Halide::saturating_cast<std::uint8_t>(c0 * srcPix0 + c1 * srcPix1 +
                                                          c2 * srcPix2 + c3 * srcPix3);

  // スケジューリングを行う。
  Halide::Var i_inner{}, j_inner{};
  dstImg.tile(i, j, i_inner, j_inner, 64, 4).vectorize(i_inner, 16).parallel(j);

  // 実行する。
  //auto output = Halide::Buffer<std::uint8_t>::make_interleaved(
  //    dstMat.ptr<std::uint8_t>(), dstMat.cols, dstMat.rows, dstMat.channels(), "output");

  Halide::Buffer<std::uint8_t> output;
  dstImg.realize(output);

  //dstImg.realize(output);
}

}  // namespace halidesample
