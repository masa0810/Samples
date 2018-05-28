
// クロック
using Clock = std::chrono::high_resolution_clock;
// 時間ポイント
using TimePoint = Clock::time_point;
// 浮動小数点秒
template <typename T>
using Seconds_ = std::chrono::duration<T>;
using Secondsf = Seconds_<float>;
using Secondsd = Seconds_<double>;
// 浮動小数点ミリ秒
template <typename T>
using MilliSeconds_ = std::chrono::duration<T, std::milli>;
using MilliSecondsf = MilliSeconds_<float>;
using MilliSecondsd = MilliSeconds_<double>;
// 浮動小数点秒取得
template <typename T>
T GetDuration(const TimePoint& s) {
  return std::chrono::duration_cast<Seconds_<T>>(Clock::now() - s).count();
}
FUNCTION_ALIAS(GetDurationf, GetDuration<float>)
FUNCTION_ALIAS(GetDurationd, GetDuration<double>)
// 浮動小数点秒取得
template <typename T>
T GetDurationMilli(const TimePoint& s) {
  return std::chrono::duration_cast<MilliSeconds_<T>>(Clock::now() - s).count();
}
FUNCTION_ALIAS(GetDurationMillif, GetDurationMilli<float>)
FUNCTION_ALIAS(GetDurationMillid, GetDurationMilli<double>)

// ベンチマーク
template <typename T, typename F>
T Benchmark(const int samples, const int iterations, const F& func) {
  auto best = std::numeric_limits<T>::infinity();
  for (auto i = 0; i < samples; i++) {
    const auto s = Clock::now();
    for (int j = 0; j < iterations; j++) func();
    auto t2 = Clock::now();
    const auto dt = GetDurationMilli<T>(s);
    if (dt < best) best = dt;
  }
  return best / iterations;
}
