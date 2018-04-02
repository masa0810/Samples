#pragma once

#include "targetver.h"

#pragma warning(push, 0)
#pragma warning(disable : 996)

// Boost
#define BOOST_ALL_DYN_LINK

// Eigen
#define MKL_DIRECT_CALL_SEQ
#define EIGEN_USE_MKL_ALL
#include <Eigen/Eigen>
#pragma comment(lib, "mkl_core.lib")
#pragma comment(lib, "mkl_intel_lp64.lib")
#pragma comment(lib, "mkl_sequential.lib")

// OpenCV
#include <opencv2/opencv.hpp>
#define OPENCV_VER "341"
#ifdef _DEBUG
#define OPENCV_SUFFIX "d"
#else
#define OPENCV_SUFFIX
#endif
#pragma comment(lib, "Halide.lib")
#pragma comment(lib, "opencv_img_hash" OPENCV_VER OPENCV_SUFFIX ".lib")
#pragma comment(lib, "opencv_world" OPENCV_VER OPENCV_SUFFIX ".lib")

// TBB
#include <tbb/tbb.h>

#pragma warning(pop)
