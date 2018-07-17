#pragma once

#include "CommonDef.h"

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <algorithm>
#include <deque>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/callable_traits/is_invocable.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>

#include <opencv2/core/types.hpp>

#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range3d.h>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/partitioner.h>
#include <tbb/scalable_allocator.h>
#include <tbb/task.h>
#include <tbb/task_group.h>

// 警告抑制解除
MSVC_WARNING_POP

namespace tbb {

#pragma region 各種型定義

/// <summary>
/// tbb_vector
/// </summary>
template <typename T>
using tbb_vector = std::vector<T, tbb::scalable_allocator<T>>;

/// <summary>
/// tbb_deque
/// </summary>
template <typename T>
using tbb_deque = std::deque<T, tbb::scalable_allocator<T>>;

/// <summary>
/// TbbAllocUnique
/// </summary>
template <typename T>
using TbbAllocUnique = commonutility::AllocUnique_<T, tbb::scalable_allocator>;

/// <summary>
/// tbb_cache_aligned_vector
/// </summary>
template <typename T>
using tbb_cache_aligned_vector = std::vector<T, tbb::cache_aligned_allocator<T>>;

/// <summary>
/// tbb_cache_aligned_deque
/// </summary>
template <typename T>
using tbb_cache_aligned_deque = std::deque<T, tbb::cache_aligned_allocator<T>>;

/// <summary>
/// TbbCacheAlignedAllocUnique
/// </summary>
template <typename T>
using TbbCacheAlignedAllocUnique = commonutility::AllocUnique_<T, tbb::cache_aligned_allocator>;

#pragma endregion

#pragma region OpenCV構造体->blocked_range

template <typename T = int>
blocked_range<T> ToRange(const cv::Range& range) {
  return {range.start, range.end};
}
template <typename T>
blocked_range2d<T> ToRange(const cv::Size_<T>& size) {
  return {0, size.height, 0, size.width};
}
template <typename T>
blocked_range2d<T> ToRange(const cv::Rect_<T>& rect) {
  return {rect.y, rect.y + rect.height, rect.x, rect.width};
}
template <typename T>
blocked_range2d<T> ToRange(const cv::Point_<T>& start, const cv::Point_<T>& end) {
  return {start.y, end.y, start.x, end.x};
}
template <typename T>
blocked_range3d<T> ToRange(const cv::Point3_<T>& start, const cv::Point3_<T>& end) {
  return {start.z, end.z, start.y, end.y, start.x, end.x};
}

#pragma endregion

#pragma region serialシリーズ

struct serial_task_group {
  template <typename F>
  void run(const F& f) {
    f();
  }
  template <typename F>
  void run_and_wait(const F& f) {
    f();
  }
  void wait() noexcept {}
  bool is_canceling() noexcept { return false; }
  void cancel() noexcept {}
};

#pragma region serial_for < Range, Body>

template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range<T>>{},
                           std::nullptr_t> = nullptr>
void serial_for(const blocked_range<T>& range, const Body& body, Args&&...) {
  body(range);
}
template <
    typename T1, typename T2, typename Body, typename... Args,
    std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T1, T2>>{},
                     std::nullptr_t> = nullptr>
void serial_for(const blocked_range2d<T1, T2>& range, const Body& body, Args&&...) {
  body(range);
}
template <typename T1, typename T2, typename T3, typename Body, typename... Args,
          std::enable_if_t<
              boost::callable_traits::is_invocable_r<void, Body, blocked_range3d<T1, T2, T3>>{},
              std::nullptr_t> = nullptr>
void serial_for(const blocked_range3d<T1, T2, T3>& range, const Body& body, Args&&...) {
  body(range);
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T>>{},
                           std::nullptr_t> = nullptr>
void serial_for(const cv::Size_<T>& range, const Body& body, Args&&...) {
  body(ToRange(range));
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T>>{},
                           std::nullptr_t> = nullptr>
void serial_for(const cv::Rect_<T>& range, const Body& body, Args&&...) {
  body(ToRange(range));
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T>>{},
                           std::nullptr_t> = nullptr>
void serial_for(const cv::Point_<T>& start, const cv::Point_<T>& end, const Body& body, Args&&...) {
  body(ToRange(start, end));
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range3d<T>>{},
                           std::nullptr_t> = nullptr>
void serial_for(const cv::Point3_<T>& start, const cv::Point3_<T>& end, const Body& body,
                Args&&...) {
  body(ToRange(start, end));
}
template <typename Body, typename... Args>
void serial_for(const cv::Range& range, const Body& body, Args&&...) {
  body(ToRange<>(range));
}

#pragma endregion

#pragma region serial_for < Index, Function>

template <typename Index, typename Index_, typename Function, typename... Args,
          std::enable_if_t<std::is_arithmetic<Index>{} && std::is_arithmetic<Index_>{} &&
                               boost::callable_traits::is_invocable_r<void, Function, Index>{},
                           std::nullptr_t> = nullptr>
void serial_for(Index first, const Index_ last, const Function& f, Args&&...) {
  const auto e = static_cast<Index>(last);
  for (; first < e; ++first) f(first);
}
template <typename Index, typename Index_1, typename Index_2, typename Function, typename... Args,
          std::enable_if_t<std::is_arithmetic<Index>{} && std::is_arithmetic<Index_1>{} &&
                               std::is_arithmetic<Index_2>{} &&
                               boost::callable_traits::is_invocable_r<void, Function, Index>{},
                           std::nullptr_t> = nullptr>
void serial_for(Index first, const Index_1 last, const Index_2 step, const Function& f, Args&&...) {
  const auto e = static_cast<Index>(last);
  const auto s = static_cast<Index>(step);
  for (; first < e; first += s) f(first);
}

#pragma endregion

#pragma region serial_for_each

template <typename Iterator, typename Function, typename... Args>
void serial_for_each(Iterator first, Iterator last, const Function& f, Args&&...) {
  std::for_each(first, last, f);
}
template <typename Range, typename Function, typename... Args>
void serial_for_each(Range& rng, const Function& f, Args&&...) {
  boost::for_each(rng, f);
}

#pragma endregion

#pragma region serial_reduce < Range, Body>

template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range<T>>{},
                           std::nullptr_t> = nullptr>
void serial_reduce(const blocked_range<T>& range, Body& body, Args&&...) {
  body(range);
}
template <
    typename T1, typename T2, typename Body, typename... Args,
    std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T1, T2>>{},
                     std::nullptr_t> = nullptr>
void serial_reduce(const blocked_range2d<T1, T2>& range, Body& body, Args&&...) {
  body(range);
}
template <typename T1, typename T2, typename T3, typename Body, typename... Args,
          std::enable_if_t<
              boost::callable_traits::is_invocable_r<void, Body, blocked_range3d<T1, T2, T3>>{},
              std::nullptr_t> = nullptr>
void serial_reduce(const blocked_range3d<T1, T2, T3>& range, Body& body, Args&&...) {
  body(range);
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T>>{},
                           std::nullptr_t> = nullptr>
void serial_reduce(const cv::Size_<T>& range, Body& body, Args&&...) {
  body(ToRange(range));
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T>>{},
                           std::nullptr_t> = nullptr>
void serial_reduce(const cv::Rect_<T>& range, Body& body, Args&&...) {
  body(ToRange(range));
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T>>{},
                           std::nullptr_t> = nullptr>
void serial_reduce(const cv::Point_<T>& start, const cv::Point_<T>& end, Body& body, Args&&...) {
  body(ToRange(start, end));
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range3d<T>>{},
                           std::nullptr_t> = nullptr>
void serial_reduce(const cv::Point3_<T>& start, const cv::Point3_<T>& end, Body& body, Args&&...) {
  body(ToRange(start, end));
}
template <typename Body, typename... Args>
void serial_reduce(const cv::Range& range, Body& body, Args&&...) {
  body(ToRange<>(range));
}

#pragma endregion

#pragma region serial_reduce < Range, Value, RealBody, Reduction>

template <typename T, typename Value, typename RealBody, typename... Args,
          std::enable_if_t<
              boost::callable_traits::is_invocable_r<Value, RealBody, blocked_range<T>, Value>{},
              std::nullptr_t> = nullptr>
Value serial_reduce(const blocked_range<T>& range, const Value& identity, const RealBody& real_body,
                    Args&&...) {
  return real_body(range, identity);
}
template <typename T1, typename T2, typename Value, typename RealBody, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<Value, RealBody,
                                                                  blocked_range2d<T1, T2>, Value>{},
                           std::nullptr_t> = nullptr>
Value serial_reduce(const blocked_range2d<T1, T2>& range, const Value& identity,
                    const RealBody& real_body, Args&&...) {
  return real_body(range, identity);
}
template <typename T1, typename T2, typename T3, typename Value, typename RealBody,
          typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<
                               Value, RealBody, blocked_range3d<T1, T2, T3>, Value>{},
                           std::nullptr_t> = nullptr>
Value serial_reduce(const blocked_range3d<T1, T2, T3>& range, const Value& identity,
                    const RealBody& real_body, Args&&...) {
  return real_body(range, identity);
}
template <typename T, typename Value, typename RealBody, typename... Args,
          std::enable_if_t<
              boost::callable_traits::is_invocable_r<Value, RealBody, blocked_range2d<T>, Value>{},
              std::nullptr_t> = nullptr>
Value serial_reduce(const cv::Size_<T>& range, const Value& identity, const RealBody& real_body,
                    Args&&...) {
  return real_body(ToRange(range), identity);
}
template <typename T, typename Value, typename RealBody, typename... Args,
          std::enable_if_t<
              boost::callable_traits::is_invocable_r<Value, RealBody, blocked_range2d<T>, Value>{},
              std::nullptr_t> = nullptr>
Value serial_reduce(const cv::Rect_<T>& range, const Value& identity, const RealBody& real_body,
                    Args&&...) {
  return real_body(ToRange(range), identity);
}
template <typename T, typename Value, typename RealBody, typename... Args,
          std::enable_if_t<
              boost::callable_traits::is_invocable_r<Value, RealBody, blocked_range2d<T>, Value>{},
              std::nullptr_t> = nullptr>
Value serial_reduce(const cv::Point_<T>& start, const cv::Point_<T>& end, const Value& identity,
                    const RealBody& real_body, Args&&...) {
  return real_body(ToRange(start, end), identity);
}
template <typename T, typename Value, typename RealBody, typename... Args,
          std::enable_if_t<
              boost::callable_traits::is_invocable_r<Value, RealBody, blocked_range3d<T>, Value>{},
              std::nullptr_t> = nullptr>
Value serial_reduce(const cv::Point3_<T>& start, const cv::Point3_<T>& end, const Value& identity,
                    const RealBody& real_body, Args&&...) {
  return real_body(ToRange(start, end), identity);
}
template <typename Value, typename RealBody, typename... Args>
Value serial_reduce(const cv::Range& range, const Value& identity, const RealBody& real_body,
                    Args&&...) {
  return real_body(ToRange<>(range), identity);
}

#pragma endregion

#pragma region serial_sort

template <typename Iterator, typename Function>
void serial_sort(Iterator first, Iterator last, const Function& f) {
  std::sort(first, last, f);
}
template <typename Range, typename Function>
void serial_sort(Range& rng, const Function& f) {
  boost::sort(rng, f);
}
template <typename Iterator>
void serial_sort(Iterator first, Iterator last) {
  std::sort(first, last);
}
template <typename Range>
void serial_sort(Range& rng) {
  boost::sort(rng);
}

#pragma endregion

#pragma endregion

#pragma region custom_parallelシリーズ

#pragma region custom_parallel_for < Range, Body>

template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range<T>>{},
                           std::nullptr_t> = nullptr>
void custom_parallel_for(const blocked_range<T>& range, const Body& body, Args&&... args) {
  parallel_for(range, body, std::forward<Args>(args)...);
}
template <
    typename T1, typename T2, typename Body, typename... Args,
    std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T1, T2>>{},
                     std::nullptr_t> = nullptr>
void custom_parallel_for(const blocked_range2d<T1, T2>& range, const Body& body, Args&&... args) {
  parallel_for(range, body, std::forward<Args>(args)...);
}
template <typename T1, typename T2, typename T3, typename Body, typename... Args,
          std::enable_if_t<
              boost::callable_traits::is_invocable_r<void, Body, blocked_range3d<T1, T2, T3>>{},
              std::nullptr_t> = nullptr>
void custom_parallel_for(const blocked_range3d<T1, T2, T3>& range, const Body& body,
                         Args&&... args) {
  parallel_for(range, body, std::forward<Args>(args)...);
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T>>{},
                           std::nullptr_t> = nullptr>
void custom_parallel_for(const cv::Size_<T>& range, const Body& body, Args&&... args) {
  parallel_for(ToRange(range), body, std::forward<Args>(args)...);
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T>>{},
                           std::nullptr_t> = nullptr>
void custom_parallel_for(const cv::Rect_<T>& range, const Body& body, Args&&... args) {
  parallel_for(ToRange(range), body, std::forward<Args>(args)...);
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T>>{},
                           std::nullptr_t> = nullptr>
void custom_parallel_for(const cv::Point_<T>& start, const cv::Point_<T>& end, const Body& body,
                         Args&&... args) {
  parallel_for(ToRange(start, end), body, std::forward<Args>(args)...);
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range3d<T>>{},
                           std::nullptr_t> = nullptr>
void custom_parallel_for(const cv::Point3_<T>& start, const cv::Point3_<T>& end, const Body& body,
                         Args&&... args) {
  parallel_for(ToRange(start, end), body, std::forward<Args>(args)...);
}
template <typename Body, typename... Args>
void custom_parallel_for(const cv::Range& range, const Body& body, Args&&... args) {
  parallel_for(ToRange<>(range), body, std::forward<Args>(args)...);
}

#pragma endregion

#pragma region custom_parallel_for < Index, Function>

template <typename Index, typename Index_, typename Function, typename... Args,
          std::enable_if_t<std::is_arithmetic<Index>{} && std::is_arithmetic<Index_>{} &&
                               boost::callable_traits::is_invocable_r<void, Function, Index>{},
                           std::nullptr_t> = nullptr>
void custom_parallel_for(Index first, const Index_ last, const Function& f, Args&&... args) {
  parallel_for(first, static_cast<Index>(last), f, std::forward<Args>(args)...);
}
template <typename Index, typename Index_1, typename Index_2, typename Function, typename... Args,
          std::enable_if_t<std::is_arithmetic<Index>{} && std::is_arithmetic<Index_1>{} &&
                               std::is_arithmetic<Index_2>{} &&
                               boost::callable_traits::is_invocable_r<void, Function, Index>{},
                           std::nullptr_t> = nullptr>
void custom_parallel_for(Index first, const Index_1 last, const Index_2 step, const Function& f,
                         Args&&... args) {
  parallel_for(first, static_cast<Index>(last), static_cast<Index>(step), f,
               std::forward<Args>(args)...);
}

#pragma endregion

#pragma region custom_parallel_reduce < Range, Body>

template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range<T>>{},
                           std::nullptr_t> = nullptr>
void custom_parallel_reduce(const blocked_range<T>& range, Body& body, Args&&... args) {
  parallel_reduce(range, body, std::forward<Args>(args)...);
}
template <
    typename T1, typename T2, typename Body, typename... Args,
    std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T1, T2>>{},
                     std::nullptr_t> = nullptr>
void custom_parallel_reduce(const blocked_range2d<T1, T2>& range, Body& body, Args&&... args) {
  parallel_reduce(range, body, std::forward<Args>(args)...);
}
template <typename T1, typename T2, typename T3, typename Body, typename... Args,
          std::enable_if_t<
              boost::callable_traits::is_invocable_r<void, Body, blocked_range3d<T1, T2, T3>>{},
              std::nullptr_t> = nullptr>
void custom_parallel_reduce(const blocked_range3d<T1, T2, T3>& range, Body& body, Args&&... args) {
  parallel_reduce(range, body, std::forward<Args>(args)...);
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T>>{},
                           std::nullptr_t> = nullptr>
void custom_parallel_reduce(const cv::Size_<T>& range, Body& body, Args&&... args) {
  parallel_reduce(ToRange(range), body, std::forward<Args>(args)...);
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T>>{},
                           std::nullptr_t> = nullptr>
void custom_parallel_reduce(const cv::Rect_<T>& range, Body& body, Args&&... args) {
  parallel_reduce(ToRange(range), body, std::forward<Args>(args)...);
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range2d<T>>{},
                           std::nullptr_t> = nullptr>
void custom_parallel_reduce(const cv::Point_<T>& start, const cv::Point_<T>& end, Body& body,
                            Args&&... args) {
  parallel_reduce(ToRange(start, end), body, std::forward<Args>(args)...);
}
template <typename T, typename Body, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<void, Body, blocked_range3d<T>>{},
                           std::nullptr_t> = nullptr>
void custom_parallel_reduce(const cv::Point3_<T>& start, const cv::Point3_<T>& end, Body& body,
                            Args&&... args) {
  parallel_reduce(ToRange(start, end), body, std::forward<Args>(args)...);
}
template <typename Body, typename... Args>
void custom_parallel_reduce(const cv::Range& range, Body& body, Args&&... args) {
  parallel_reduce(ToRange<>(range), body, std::forward<Args>(args)...);
}

#pragma endregion

#pragma region custom_parallel_reduce < Range, Value, RealBody, Reduction>

template <typename T, typename Value, typename RealBody, typename... Args,
          std::enable_if_t<
              boost::callable_traits::is_invocable_r<Value, RealBody, blocked_range<T>, Value>{},
              std::nullptr_t> = nullptr>
Value custom_parallel_reduce(const blocked_range<T>& range, const Value& identity,
                             const RealBody& real_body, Args&&... args) {
  return parallel_reduce(range, identity, real_body, std::forward<Args>(args)...);
}
template <typename T1, typename T2, typename Value, typename RealBody, typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<Value, RealBody,
                                                                  blocked_range2d<T1, T2>, Value>{},
                           std::nullptr_t> = nullptr>
Value custom_parallel_reduce(const blocked_range2d<T1, T2>& range, const Value& identity,
                             const RealBody& real_body, Args&&... args) {
  return parallel_reduce(range, identity, real_body, std::forward<Args>(args)...);
}
template <typename T1, typename T2, typename T3, typename Value, typename RealBody,
          typename... Args,
          std::enable_if_t<boost::callable_traits::is_invocable_r<
                               Value, RealBody, blocked_range3d<T1, T2, T3>, Value>{},
                           std::nullptr_t> = nullptr>
Value custom_parallel_reduce(const blocked_range3d<T1, T2, T3>& range, const Value& identity,
                             const RealBody& real_body, Args&&... args) {
  return parallel_reduce(range, identity, real_body, std::forward<Args>(args)...);
}
template <typename T, typename Value, typename RealBody, typename... Args,
          std::enable_if_t<
              boost::callable_traits::is_invocable_r<Value, RealBody, blocked_range2d<T>, Value>{},
              std::nullptr_t> = nullptr>
Value custom_parallel_reduce(const cv::Size_<T>& range, const Value& identity,
                             const RealBody& real_body, Args&&... args) {
  return parallel_reduce(ToRange(range), identity, real_body, std::forward<Args>(args)...);
}
template <typename T, typename Value, typename RealBody, typename... Args,
          std::enable_if_t<
              boost::callable_traits::is_invocable_r<Value, RealBody, blocked_range2d<T>, Value>{},
              std::nullptr_t> = nullptr>
Value custom_parallel_reduce(const cv::Rect_<T>& range, const Value& identity,
                             const RealBody& real_body, Args&&... args) {
  return parallel_reduce(ToRange(range), identity, real_body, std::forward<Args>(args)...);
}
template <typename T, typename Value, typename RealBody, typename... Args,
          std::enable_if_t<
              boost::callable_traits::is_invocable_r<Value, RealBody, blocked_range2d<T>, Value>{},
              std::nullptr_t> = nullptr>
Value custom_parallel_reduce(const cv::Point_<T>& start, const cv::Point_<T>& end,
                             const Value& identity, const RealBody& real_body, Args&&... args) {
  return parallel_reduce(ToRange(start, end), identity, real_body, std::forward<Args>(args)...);
}
template <typename T, typename Value, typename RealBody, typename... Args,
          std::enable_if_t<
              boost::callable_traits::is_invocable_r<Value, RealBody, blocked_range3d<T>, Value>{},
              std::nullptr_t> = nullptr>
Value custom_parallel_reduce(const cv::Point3_<T>& start, const cv::Point3_<T>& end,
                             const Value& identity, const RealBody& real_body, Args&&... args) {
  return parallel_reduce(ToRange(start, end), identity, real_body, std::forward<Args>(args)...);
}
template <typename Value, typename RealBody, typename... Args>
Value custom_parallel_reduce(const cv::Range& range, const Value& identity,
                             const RealBody& real_body, Args&&... args) {
  return parallel_reduce(ToRange<>(range), identity, real_body, std::forward<Args>(args)...);
}

#pragma endregion

#pragma endregion

#pragma region task_groupのparallel・serial切り替え

#ifdef TBB_NO_PARALLEL
using tbb_task_group = serial_task_group;
FUNCTION_ALIAS(tbb_for, serial_for)
FUNCTION_ALIAS(tbb_for_each, serial_for_each)
FUNCTION_ALIAS(tbb_reduce, serial_reducereduce)
FUNCTION_ALIAS(tbb_sort, serial_sort)
#else
using tbb_task_group = task_group;
FUNCTION_ALIAS(tbb_for, custom_parallel_for)
FUNCTION_ALIAS(tbb_for_each, parallel_for_each)
FUNCTION_ALIAS(tbb_reduce, custom_parallel_reduce)
FUNCTION_ALIAS(tbb_sort, parallel_sort)
#endif

#pragma endregion

#pragma region ifシリーズ

#pragma region if定義マクロ

#ifdef TBB_NO_PARALLEL
// 並列化OFF
#define __DEFINED_TBB_IF_IMPL(name, para, seri) \
  template <typename... Args>                   \
  void name(const bool, Args&&... args) {       \
    seri(std::forward<Args>(args)...);          \
  }
#elif defined(TBB_FORCE_PARALLEL)
// 強制並列化
#define __DEFINED_TBB_IF_IMPL(name, para, seri) \
  template <typename... Args>                   \
  void name(const bool, Args&&... args) {       \
    para(std::forward<Args>(args)...);          \
  }
#else
#define __DEFINED_TBB_IF_IMPL(name, para, seri) \
  template <typename... Args>                   \
  void name(const bool flag, Args&&... args) {  \
    if (flag)                                   \
      para(std::forward<Args>(args)...);        \
    else                                        \
      seri(std::forward<Args>(args)...);        \
  }
#endif

#ifdef TBB_NO_PARALLEL
#define __DEFINED_TBB_IF(name, para, seri) \
  __DEFINED_TBB_IF_IMPL(name, para, seri)  \
  template <bool Flag, typename... Args>   \
  void name(Args&&... args) {              \
    seri(std::forward<Args>(args)...);     \
  }
#else
#define __DEFINED_TBB_IF(name, para, seri)                                                  \
  __DEFINED_TBB_IF_IMPL(name, para, seri)                                                   \
  template <bool Flag, typename... Args, std::enable_if_t<Flag, std::nullptr_t> = nullptr>  \
  void name(Args&&... args) {                                                               \
    para(std::forward<Args>(args)...);                                                      \
  }                                                                                         \
  template <bool Flag, typename... Args, std::enable_if_t<!Flag, std::nullptr_t> = nullptr> \
  void name(Args&&... args) {                                                               \
    seri(std::forward<Args>(args)...);                                                      \
  }
#endif

#ifdef TBB_NO_PARALLEL
#define __DEFINED_TBB_REDUCE_IF_IMPL(name, para, seri)                                 \
  template <typename... Args>                                                          \
  auto name(const bool, Args&&... args)->decltype(para(std::forward<Args>(args)...)) { \
    return seri(std::forward<Args>(args)...);                                          \
  }
#elif defined(TBB_FORCE_PARALLEL)
#define __DEFINED_TBB_REDUCE_IF_IMPL(name, para, seri)                                 \
  template <typename... Args>                                                          \
  auto name(const bool, Args&&... args)->decltype(para(std::forward<Args>(args)...)) { \
    return para(std::forward<Args>(args)...);                                          \
  }
#else
#define __DEFINED_TBB_REDUCE_IF_IMPL(name, para, seri)                                      \
  template <typename... Args>                                                               \
  auto name(const bool flag, Args&&... args)->decltype(para(std::forward<Args>(args)...)) { \
    if (flag)                                                                               \
      return para(std::forward<Args>(args)...);                                             \
    else                                                                                    \
      return seri(std::forward<Args>(args)...);                                             \
  }
#endif

#ifdef TBB_NO_PARALLEL
#define __DEFINED_TBB_REDUCE_IF(name, para, seri)                          \
  __DEFINED_TBB_REDUCE_IF_IMPL(name, para, seri)                           \
  template <bool Flag, typename... Args>                                   \
  auto name(Args&&... args)->decltype(seri(std::forward<Args>(args)...)) { \
    return seri(std::forward<Args>(args)...);                              \
  }
#else
#define __DEFINED_TBB_REDUCE_IF(name, para, seri)                                           \
  __DEFINED_TBB_REDUCE_IF_IMPL(name, para, seri)                                            \
  template <bool Flag, typename... Args, std::enable_if_t<Flag, std::nullptr_t> = nullptr>  \
  auto name(Args&&... args)->decltype(para(std::forward<Args>(args)...)) {                  \
    return para(std::forward<Args>(args)...);                                               \
  }                                                                                         \
  template <bool Flag, typename... Args, std::enable_if_t<!Flag, std::nullptr_t> = nullptr> \
  auto name(Args&&... args)->decltype(seri(std::forward<Args>(args)...)) {                  \
    return seri(std::forward<Args>(args)...);                                               \
  }
#endif

#pragma endregion

// tbb_for_if
__DEFINED_TBB_IF(tbb_for_if, custom_parallel_for, serial_for)

// tbb_for_each_if
__DEFINED_TBB_IF(tbb_for_each_if, parallel_for_each, serial_for_each)

// tbb_reduce_if
__DEFINED_TBB_REDUCE_IF(tbb_reduce_if, custom_parallel_reduce, serial_reduce)

// tbb_sort_if
__DEFINED_TBB_IF(tbb_sort_if, parallel_sort, serial_sort)

#pragma endregion

}  // namespace tbb

// シリアライズ設定
namespace boost { namespace serialization {

template <class Archive, typename T, typename Allocator>
void save(Archive& ar, const tbb::concurrent_vector<T, Allocator>& t, const unsigned) {
  const auto size = t.size();
  ar& make_nvp("Size", size);
  ar& make_array(&t[0], t.size());
}
template <class Archive, typename T, typename Allocator>
void load(Archive& ar, tbb::concurrent_vector<T, Allocator>& t, const unsigned) {
  typename tbb::concurrent_vector<T, Allocator>::size_type size;
  ar& make_nvp("Size", size);
  if (t.size != size) t.grow_to_at_least(size);
  ar& make_array(&t[0], t.size());
}
template <class Archive, typename T, typename Allocator>
void serialize(Archive& ar, tbb::concurrent_vector<T, Allocator>& t, const unsigned version) {
  split_free(ar, t, version);
}

}}  // namespace boost::serialization
