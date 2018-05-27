#pragma once

#pragma warning(push, 0)

// Windows
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "targetver.h"

// Boost
#define BOOST_ALL_DYN_LINK

// Eigen
#define MKL_DIRECT_CALL_SEQ
#define EIGEN_USE_MKL_ALL
#include <Eigen/Eigen>
#pragma comment(lib, "mkl_core.lib")
#pragma comment(lib, "mkl_intel_lp64.lib")
#pragma comment(lib, "mkl_sequential.lib")

// fmt
#include <fmt/format.h>
#ifdef _DEBUG
#define FMT_SUFFIX "d"
#else
#define FMT_SUFFIX
#endif
#pragma comment(lib, "fmt" FMT_SUFFIX ".lib")

// Halide
#include <Halide.h>
#pragma comment(lib, "Halide.lib")

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

// TBB
#include <tbb/tbb.h>

#pragma warning(pop)