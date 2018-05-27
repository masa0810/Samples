#pragma once

#pragma warning(push, 0)

#include <opencv2/core.hpp>

#pragma warning(pop)

namespace rawsample {

void ResizeWithRawAccess(const cv::Mat& src_image, cv::Mat& dst_image);

}
