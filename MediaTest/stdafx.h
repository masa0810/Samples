#pragma once

#include "targetver.h"

// Eigen
#define EIGEN_USE_MKL_ALL
#include <Eigen/Eigen>

// OpenCV
#include <opencv2/opencv.hpp>
#define OPENCV_VER "341"
#ifdef _DEBUG
#define OPENCV_SUFFIX "d"
#else
#define OPENCV_SUFFIX
#endif
#pragma comment(lib, "opencv_img_hash" OPENCV_VER OPENCV_SUFFIX ".lib")
#pragma comment(lib, "opencv_world" OPENCV_VER OPENCV_SUFFIX ".lib")
#pragma comment(lib, "Halide.lib")

// TBB
#include <tbb/tbb.h>
