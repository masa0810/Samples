#ifdef __clang__
#pragma clang diagnostic ignored "-Wpragma-once-outside-header"
#endif
#pragma once

//-------------------------------------------
// コンパイラ別マクロ
//-------------------------------------------

// Clangの場合
#ifdef __clang__
#define CLANG_DIAGNOSTIC_PUSH __pragma(clang diagnostic push)
#define CLANG_DIAGNOSTIC_POP __pragma(clang diagnostic pop)
#define CLANG_DIAGNOSTIC_WARNING(name) __pragma(clang diagnostic warning name)
#define CLANG_DIAGNOSTIC_IGNORED(name) __pragma(clang diagnostic ignored name)
#else
#define CLANG_DIAGNOSTIC_PUSH
#define CLANG_DIAGNOSTIC_POP
#define CLANG_DIAGNOSTIC_WARNING(name)
#define CLANG_DIAGNOSTIC_IGNORED(name)
#endif

// MSVCの場合
#ifdef _MSC_VER
#define MSVC_WARNING_PUSH __pragma(warning(push))
#define MSVC_WARNING_POP                                                          \
  __pragma(warning(pop)) __pragma(warning(default : 244 503 702 706 714 723 724)) \
      __pragma(warning(default                                                    \
                       : ALL_CODE_ANALYSIS_WARNINGS))
#define MSVC_WARNING_ENABLE(num) __pragma(warning(enable : num))
#define MSVC_WARNING_DISABLE(num) __pragma(warning(disable : num))
#define MSVC_WARNING_DEFAULT(num) __pragma(warning(default : num))
#define MSVC_ALL_WARNING_PUSH                                                         \
  __pragma(warning(push, 0)) __pragma(warning(disable : 244 503 702 706 714 723 724)) \
      __pragma(warning(disable                                                        \
                       : ALL_CODE_ANALYSIS_WARNINGS))
#else
#define MSVC_WARNING_PUSH
#define MSVC_WARNING_POP
#define MSVC_WARNING_ENABLE(num)
#define MSVC_WARNING_DISABLE(num)
#define MSVC_WARNING_DEFAULT(num)
#define MSVC_ALL_WARNING_PUSH
#endif

//-------------------------------------------
// 各種設定
//-------------------------------------------

#ifdef _MSC_VER
// Windows.hで"min"と"max"をdefineしない(C++標準ライブラリの"min"と"max"を使う)
#define NOMINMAX
// 頻繁に使用されないAPIを使用しない
#define VC_EXTRALEAN
// 頻繁に使用されないAPIを使用しない
#define WIN32_LEAN_AND_MEAN
// GDI系APIをOFF
#define NOGDI
#endif

// Debug設定
#ifdef _DEBUG
// Benchmarkのデバッグ版使用
#define BENCHMARK_USE_DEBUG 1
// GoogleTestのデバッグ版使用
#define GTEST_USE_DEBUG 1
// OpenCVのデバッグ版使用
#define OPENCV_USE_DEBUG 1
// TBBのデバッグ版使用
#define TBB_USE_DEBUG 1
#define TBB_DISABLE_PROXY 1

// TBBの並列化無効
//#define TBB_NO_PARALLEL 1
// アサート有効
#define ENABLE_ASSERT 1
#endif

// 動的リンク設定
#ifdef DYN_LINK
// Boostの動的リンク有効
#define BOOST_ALL_DYN_LINK 1
#endif

#if !defined(DYN_LINK) && !defined(_DEBUG)
// 強制並列化
#define TBB_FORCE_PARALLEL 1
// switch文のdefault最適化
#define ENABLE_NO_DEFAULT 1
// ログ出力OFF
//#define DISABLE_LOG 1
#endif

#ifdef _MSC_VER

//-------------------------------------------
// 設定ヘッダインクルード
//-------------------------------------------

#include <CodeAnalysis/Warnings.h>

// SDK version
#include "targetver.h"

#define BENCHMARK_USE_MAIN
#include <benchmarkset.h>
//#define GTEST_USE_MAIN
//#include <gtestset.h>

#include <fmtset.h>
#include <opencvset.h>
#include <tbbset.h>

// C4996警告対策
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#if defined(_DEBUG) && !defined(_SCL_SECURE_NO_WARNINGS)
#define _SCL_SECURE_NO_WARNINGS
#endif
#include <cstdio>
#ifdef _SCL_SECURE_NO_WARNINGS
#undef _SCL_SECURE_NO_WARNINGS
#endif
#undef _CRT_SECURE_NO_WARNINGS
#endif

//-------------------------------------------
// ライブラリインクルード
//-------------------------------------------

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

// Standard C++

// Windows Library

// Boost

// Eigen

// OpenCV

// TBB

// 警告抑制解除
MSVC_WARNING_POP

//-------------------------------------------
// 特殊設定
//-------------------------------------------

// デバッグビルドに限り未使用変数を許可
#ifdef _DEBUG
MSVC_WARNING_DISABLE(100 189)
#endif

#define _EXPORT_IMGPROC_

//-------------------------------------------
// stdafx.cpp用警告対策
//-------------------------------------------

#endif
