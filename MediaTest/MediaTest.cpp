#pragma warning(push, 0)

#include <cstdint>

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#pragma warning(pop)

int main() {
  cv::VideoCapture cap(R"(C:\Library\Current\Source\opencv\samples\data\vtest.avi)");
  const cv::Size capSize(cvFloor(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                         cvFloor(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

  cv::VideoWriter wri("mjpeg.avi", cv::CAP_GSTREAMER, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                      5.0, {1, 1});
  if (cap.isOpened()) {
    cv::Mat src;
    std::vector<std::uint8_t> buf;

    while (cap.read(src)) {
      cv::imshow("Debug", src);

      cv::imencode(".jpg", src, buf);
      wri.write(cv::Mat(buf));

      cv::waitKey(2);
    }
  }
  return 0;
}
