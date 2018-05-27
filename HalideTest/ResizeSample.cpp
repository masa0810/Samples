#include "ResizeSample.h"

#pragma warning(push, 0)

#include <cstdint>

#pragma warning(pop)

namespace rawsample {

const std::uint8_t* getPixel(const std::uint8_t* data, const int row,
                             const int col, const std::size_t step,
                             const std::size_t elem_size) {
  return data + (row * step + col * elem_size);
}

std::uint8_t interpolate(const float c0, const float c1, const float c2,
                         const float c3, const std::uint8_t* p0,
                         const std::uint8_t* p1, const std::uint8_t* p2,
                         const std::uint8_t* p3, const int i) {
  return cv::saturate_cast<std::uint8_t>(c0 * p0[i] + c1 * p1[i] + c2 * p2[i] +
                                         c3 * p3[i]);
}

void ResizeWithRawAccess(const cv::Mat& src_image, cv::Mat& dst_image) {
  // 入力画像の各種値・ポインタを取り出す。
  const int src_cols = src_image.cols;
  const int src_rows = src_image.rows;
  const std::size_t src_step = src_image.step;
  const std::size_t src_elem_size = src_image.elemSize();
  const std::uint8_t* src_data = src_image.data;

  // 出力画像の各種値・ポインタを取り出す。
  const int dst_cols = dst_image.cols;
  const int dst_rows = dst_image.rows;
  const auto dst_step = static_cast<int>(dst_image.step);
  const auto dst_elem_size = static_cast<int>(dst_image.elemSize());
  std::uint8_t* dst_data = dst_image.data;

  // 拡大縮小率の逆数を計算する。
  const float sc = static_cast<float>(src_cols) / dst_cols;
  const float sr = static_cast<float>(src_rows) / dst_rows;

  // 拡大縮小後の画素を左上から順に走査する。
  // y軸方向の走査
  for (auto j = 0, jd = 0; j < dst_rows; ++j, jd += dst_step) {
    // 元画像における位置yを算出する。
    const float fj = j * sr;
    const int cj0 = static_cast<int>(fj);  // 端数を切り捨てる。
    const float dj = fj - cj0;

    // +1した値が画像内に収まるように調整する。
    const int cj1 = cj0 + 1 >= src_rows ? cj0 : cj0 + 1;

    // 拡大縮小後画像へのポインタの位置を更新する。
    std::uint8_t* dst_p = dst_data + jd;

    // x軸方向の走査
    for (auto i = 0, id = 0; i < dst_cols; ++i, id += dst_elem_size) {
      // 元画像における位置xを算出する。
      const float fi = i * sc;
      const int ci0 = static_cast<int>(fi);
      const float di = fi - ci0;  // dx

      const int ci1 = ci0 + 1 >= src_cols ? ci0 : ci0 + 1;

      // 面積を計算する。
      const float c0 = (1.0f - dj) * (1.0f - di);
      const float c1 = (1.0f - dj) * di;
      const float c2 = dj * (1.0f - di);
      const float c3 = dj * di;

      // 周辺画素を取り出す。
      const std::uint8_t* src_pixel0 =
          getPixel(src_data, cj0, ci0, src_step, src_elem_size);
      const std::uint8_t* src_pixel1 =
          getPixel(src_data, cj0, ci1, src_step, src_elem_size);
      const std::uint8_t* src_pixel2 =
          getPixel(src_data, cj1, ci0, src_step, src_elem_size);
      const std::uint8_t* src_pixel3 =
          getPixel(src_data, cj1, ci1, src_step, src_elem_size);

      // ポインタ位置を更新する。
      std::uint8_t* dst_pixel = dst_p + id;

      // RGB値を計算する。
      dst_pixel[0] = interpolate(c0, c1, c2, c3, src_pixel0, src_pixel1,
                                 src_pixel2, src_pixel3, 0);
      dst_pixel[1] = interpolate(c0, c1, c2, c3, src_pixel0, src_pixel1,
                                 src_pixel2, src_pixel3, 1);
      dst_pixel[2] = interpolate(c0, c1, c2, c3, src_pixel0, src_pixel1,
                                 src_pixel2, src_pixel3, 2);
    }
  }
}

}  // namespace rawsample
