#pragma once

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <cstdlib>

#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

#ifdef _MSC_VER
#include <Windows.h>
#endif

#include <boost/current_function.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/singleton.hpp>
#include <boost/serialization/split_member.hpp>

// 警告抑制解除
MSVC_WARNING_POP

#ifdef _MSC_VER
// ファイル名&行番号マクロ
#define __FILE_LINE__ __FILE__ "(" BOOST_PP_STRINGIZE(__LINE__) ")"
// ビルドメッセージヘッダ生成マクロ
#define __MSG_HEADER__(type) __FILE_LINE__ " : " type " : "
// ビルドメッセージタイプ(エラー)
#define __ERR__ __MSG_HEADER__("error")
// ビルドメッセージタイプ(警告)
#define __WARN__ __MSG_HEADER__("warning")
// ビルドメッセージタイプ(メッセージ)
#define __MSG__ __MSG_HEADER__("message")
// ビルドメッセージタイプ(ToDo)
#define __TODO__ __MSG_HEADER__("to do")
// ビルドメッセージマクロ
#define BUILD_MESSAGE(type, msg) __pragma(message(type msg))
#else
// ファイル名&行番号マクロ
#define __FILE_LINE__
// ビルドメッセージヘッダ生成マクロ
#define __MSG_HEADER__(type)
// ビルドメッセージタイプ(エラー)
#define __ERR__
// ビルドメッセージタイプ(警告)
#define __WARN__
// ビルドメッセージタイプ(メッセージ)
#define __MSG__
// ビルドメッセージタイプ(ToDo)
#define __TODO__
// ビルドメッセージマクロ
#define BUILD_MESSAGE(type, msg)
#endif

// 関数エイリアス
#define FUNCTION_ALIAS(name, target)                                                \
  template <typename... Args>                                                       \
  inline auto name(Args&&... args)->decltype(target(std::forward<Args>(args)...)) { \
    return target(std::forward<Args>(args)...);                                     \
  }

/// <summary>
/// 共通ユーティリティのネームスペース
/// </summary>
namespace commonutility {
/// <summary>
/// 詳細実装のネームスペース
/// </summary>
namespace detail {
/// <summary>
/// アサート構造体
/// </summary>
struct AssertArg {
  //! アサートフラグ
  bool Flag = false;
  //! 判定式
  std::string Expression = {};
  //! メッセージ
  std::string Message = {};

  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  AssertArg() = default;

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  AssertArg(const AssertArg&) = default;

  /// <summary>
  /// 代入演算子
  /// </summary>
  AssertArg& operator=(const AssertArg&) = default;

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  AssertArg(AssertArg&&) = default;

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  AssertArg& operator=(AssertArg&&) = default;

  /// <summary>
  /// デストラクタ
  /// </summary>
  virtual ~AssertArg() = default;

  /// <summary>
  /// コンストラクタ
  /// </summary>
  /// <param name="flag">[in] アサートフラグ</param>
  /// <param name="expr">[in] 判定式</param>
  /// <param name="msg">[in] メッセージ</param>
  AssertArg(const bool flag, const std::string& expr = {}, const std::string& msg = {})
      : Flag(flag), Expression(expr), Message(msg) {}
};

/// <summary>
/// アサート引数解析
/// </summary>
/// <param name="c">[in] アサートフラグ</param>
/// <param name="msg">[in] メッセージ</param>
/// <param name="ptrArgs">[in] アサートマクロ引数文字列</param>
inline AssertArg GetArg(const bool c, const std::string& msg, const char* const ptrArgs) {
  const std::string args(ptrArgs);
  auto flag = true;
  auto i = 0;
  const auto strSize = static_cast<int>(args.size());
  for (; i < strSize && (!flag || args[i] != ','); ++i) {
    if (args[i] == '"') flag = !flag;
  }
  for (--i; i > 0 && args[i] == ' '; --i)
    ;
  return {!c, args.substr(0, i + 1), msg};
}

/// <summary>
/// アサート引数解析
/// </summary>
/// <param name="c">[in] アサートフラグ</param>
/// <param name="msg">[in] メッセージ</param>
/// <param name="ptrArgs">[in] アサートマクロ引数文字列</param>
inline AssertArg GetArg(const bool c, const char* const msg, const char* const ptrArgs) {
  return GetArg(c, std::string(msg), ptrArgs);
}

/// <summary>
/// アサート引数解析
/// </summary>
/// <param name="c">[in] アサートフラグ</param>
/// <param name="ptrArgs">[in] アサートマクロ引数文字列</param>
inline AssertArg GetArg(const bool c, const char* const ptrArgs) {
  return {!c, std::string(ptrArgs), std::string()};
}

/// <summary>
/// アサート引数解析
/// </summary>
/// <param name="msg">[in] メッセージ</param>
inline AssertArg GetArg(const char* const msg, const char* const) {
  return {true, std::string(), std::string(msg)};
}

/// <summary>
/// アサート引数解析
/// </summary>
/// <param name="msg">[in] メッセージ</param>
inline AssertArg GetArg(const std::string& msg, const char* const) {
  return {true, std::string(), msg};
}
}  // namespace detail
}  // namespace commonutility

// アサート未定義対応
#ifndef Assert
#ifdef ENABLE_ASSERT
#ifdef _MSC_VER
#define _SET_ABORT_BEHAVIOR(x, y) _set_abort_behavior(x, y);
#else
#define _SET_ABORT_BEHAVIOR(...)
#endif
#define Assert(...)                                                                             \
  {                                                                                             \
    const auto arg = commonutility::detail::GetArg(__VA_ARGS__, #__VA_ARGS__);                  \
    if (arg.Flag) {                                                                             \
      _SET_ABORT_BEHAVIOR(0, _WRITE_ABORT_MSG)                                                  \
      std::cerr << "!!!!!!!!!! Assert !!!!!!!!!!" << std::endl;                                 \
      if (!arg.Expression.empty()) std::cerr << "Expression : " << arg.Expression << std::endl; \
      if (!arg.Message.empty()) std::cerr << "Message : " << arg.Message << std::endl;          \
      std::cerr << "Function : " << BOOST_CURRENT_FUNCTION << std::endl;                        \
      std::cerr << "File : " << __FILE__ << std::endl;                                          \
      std::cerr << "Line : " << __LINE__ << std::endl;                                          \
      std::system("pause");                                                                     \
      std::abort();                                                                             \
    }                                                                                           \
  }
#else
#ifndef MSVC_WARNING_DISABLE_PUSH
#define MSVC_WARNING_DISABLE_PUSH(n) \
  MSVC_WARNING_PUSH                  \
  MSVC_WARNING_DISABLE(n)
#endif

#define Assert(...)                                           \
  MSVC_WARNING_DISABLE_PUSH(127)                              \
  while (false) {                                             \
    MSVC_WARNING_POP                                          \
    commonutility::detail::GetArg(__VA_ARGS__, #__VA_ARGS__); \
  }
#endif
#endif

// swithc文のdefault最適化マクロの定義
#if defined(ENABLE_NO_DEFAULT) && defined(_MSC_VER)
#define NoDefault(...)      \
  MSVC_WARNING_DISABLE(702) \
  __assume(0)
#else
#define NoDefault Assert
#endif

#ifdef DYN_LINK

namespace commonutility { namespace detail {
template <typename F = std::function<void()>>
class InitFunctions_ {
  std::mutex m_mutex = {};
  std::unordered_map<std::string, F> m_functions = {};

  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  InitFunctions_() = default;

 public:
#pragma region デフォルトメソッド定義

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  InitFunctions_(const InitFunctions_&) = delete;

  /// <summary>
  /// 代入演算子
  /// </summary>
  InitFunctions_& operator=(const InitFunctions_&) = default;

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  InitFunctions_(InitFunctions_&&) = delete;

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  InitFunctions_& operator=(InitFunctions_&&) = default;

  /// <summary>
  /// デストラクタ
  /// </summary>
  virtual ~InitFunctions_() = default;

#pragma endregion

  void Set(const std::string& title, F&& func) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_functions.find(title) == std::end(m_functions)) m_functions.emplace(title, func);
  }

  const std::unordered_map<std::string, F>& Get() const { return m_functions; }

  // シングルトン
  static InitFunctions_& Singleton() {
    static InitFunctions_ instance;
    return instance;
  }
};
using InitFunctions = InitFunctions_<>;
}}  // namespace commonutility::detail

#endif

// シングルトン初期化マクロ設定
#ifndef SINGLETON_INIT
#if defined(_LIB)
#define SINGLETON_INIT(X)
#elif defined(DYN_LINK)
#define SINGLETON_INIT(X)                                                       \
  namespace X##_initnamespace {                                                 \
    struct X##_initclass {                                                      \
      X##_initclass() {                                                         \
        commonutility::detail::InitFunctions::Singleton().Set(#X, [] { (X); }); \
      }                                                                         \
      static const X##_initclass& Singleton() {                                 \
        return boost::serialization::singleton<                                 \
            X##_initnamespace::X##_initclass>::get_const_instance();            \
      }                                                                         \
    };                                                                          \
  }
#else
#define SINGLETON_INIT(X)                                            \
  namespace X##_initnamespace {                                      \
    struct X##_initclass {                                           \
      X##_initclass() {                                              \
        std::cout << "singleton initilize : " << #X << std::endl;    \
        (X);                                                         \
      }                                                              \
      static const X##_initclass& Singleton() {                      \
        return boost::serialization::singleton<                      \
            X##_initnamespace::X##_initclass>::get_const_instance(); \
      }                                                              \
    };                                                               \
  }
#endif
#endif

#ifndef SINGLETON_INITIALIZER
#if defined(DYN_LINK)
#define SINGLETON_INITIALIZER()                                                         \
  for (const auto& keyPair : commonutility::detail::InitFunctions::Singleton().Get()) { \
    std::cout << "singleton initilizer : " << keyPair.first << std::endl;               \
    keyPair.second();                                                                   \
  }
#else
#define SINGLETON_INITIALIZER()
#endif
#endif

namespace commonutility { namespace detail {
template <typename T = std::tuple<const char* const, int>>
class Resources_ {
  std::unordered_map<std::string, T> m_resources = {};

  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  Resources_() = default;

 public:
#pragma region デフォルトメソッド定義

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  Resources_(const Resources_&) = delete;

  /// <summary>
  /// 代入演算子
  /// </summary>
  Resources_& operator=(const Resources_&) = default;

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  Resources_(Resources_&&) = delete;

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  Resources_& operator=(Resources_&&) = default;

  /// <summary>
  /// デストラクタ
  /// </summary>
  virtual ~Resources_() = default;

#pragma endregion

  void Set(const std::unordered_map<std::string, T>& resources) { m_resources = resources; }

  const std::unordered_map<std::string, T>& Get() const { return m_resources; }

  // シングルトン
  static Resources_& Singleton() {
    static Resources_ instance;
    return instance;
  }
};
using Resources = Resources_<>;
}}  // namespace commonutility::detail

#ifndef RESOURCE_SET
#define RESOURCE_SET(x) commonutility::detail::Resources::Singleton().Set(x)
#endif

#ifndef RESOURCE_GET
#define RESOURCE_GET() commonutility::detail::Resources::Singleton().Get()
#endif

/// <summary>
/// スコープ付きenum作成マクロ
/// </summary>
/// <param name="NS">enumを定義するネームスペース名</param>
/// <param name="NAME">生成するenumの型名</param>
/// <param name="SEQ">生成するenumの値</param>
#define ENUM_CLASS_CREATOR(NAME, SEQ) enum class NAME { BOOST_PP_SEQ_ENUM(SEQ) };

/// <summary>
/// stream出力用case作成マクロ
/// </summary>
/// <param name="R">未使用</param>
/// <param name="NAME">生成するenumの型名</param>
/// <param name="X">生成するenumの値</param>
/// <remarks>ENUM_OSTREAM_CREATOR マクロの中で使用する実装マクロ</remarks>
#define OSTREAMABLE_ENUM_CLASS_CASE_IMPL(R, NAME, X) \
  case NAME::X:                                      \
    os << BOOST_PP_STRINGIZE(X);                     \
    break;

/// <summary>
/// stream出力関数作成マクロ
/// </summary>
/// <param name="NS">enumを定義するネームスペース名</param>
/// <param name="NAME">生成するenumの型名</param>
/// <param name="SEQ">生成するenumの値</param>
/// <remarks>enumを文字列に変換して出力するストリームメソッドを作成する。</remarks>
#define ENUM_OSTREAM_CREATOR(NAME, SEQ)                                                \
  template <typename charT, typename traits>                                           \
  std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os, \
                                                const NAME& e) {                       \
    switch (e) {                                                                       \
      BOOST_PP_SEQ_FOR_EACH(OSTREAMABLE_ENUM_CLASS_CASE_IMPL, NAME, SEQ)               \
      default:                                                                         \
        NoDefault("Does not have a corresponding string");                             \
        break;                                                                         \
    }                                                                                  \
    return os;                                                                         \
  }

/// <summary>
/// stream入力用if-else作成マクロ
/// </summary>
/// <param name="R">未使用</param>
/// <param name="NAME">生成するenumの型名</param>
/// <param name="X">生成するenumの値</param>
/// <remarks>ENUM_ISTREAM_CREATOR マクロの中で使用する実装マクロ</remarks>
#define ISTREAMABLE_ENUM_CLASS_IF_ELSE_IMPL(R, NAME, X) \
  if (!tmp.compare(BOOST_PP_STRINGIZE(X))) {            \
    e = NAME::X;                                        \
  } else

/// <summary>
/// stream入力関数作成マクロ
/// </summary>
/// <param name="NS">enumを定義するネームスペース名</param>
/// <param name="NAME">生成するenumの型名</param>
/// <param name="SEQ">生成するenumの値</param>
/// <remarks>入力文字列からenumに変換するストリームメソッドを作成する。</remarks>
#define ENUM_ISTREAM_CREATOR(NAME, SEQ)                                                           \
  template <typename charT, typename traits>                                                      \
  std::basic_istream<charT, traits>& operator>>(std::basic_istream<charT, traits>& is, NAME& e) { \
    std::basic_string<charT, traits> tmp;                                                         \
    std::getline(is, tmp);                                                                        \
    BOOST_PP_SEQ_FOR_EACH(ISTREAMABLE_ENUM_CLASS_IF_ELSE_IMPL, NAME, SEQ) {                       \
      Assert("Does not have a corresponding enum");                                               \
    }                                                                                             \
    return is;                                                                                    \
  }

/// <summary>
/// stream作成マクロ
/// </summary>
/// <param name="NS">enumを定義するネームスペース名</param>
/// <param name="NAME">生成するenumの型名</param>
/// <param name="SEQ">生成するenumの値</param>
/// <remarks>
/// 入出力用のstremメソッドを定義し、シリアライズ時に
/// strig形式を使用するためのラッパークラスも同時に定義する。
/// </remarks>
#define ENUM_STREAM_CREATOR(NAME, SEQ)                           \
  ENUM_OSTREAM_CREATOR(NAME, SEQ)                                \
  ENUM_ISTREAM_CREATOR(NAME, SEQ)                                \
  namespace NAME##_string_out {                                  \
    template <typename enumType>                                 \
    class NAME##Clone {                                          \
      NAME& m_E;                                                 \
      friend class boost::serialization::access;                 \
      BOOST_SERIALIZATION_SPLIT_MEMBER()                         \
      template <typename Archive>                                \
      void save(Archive& ar, const unsigned) const {             \
        std::string tmp = boost::lexical_cast<std::string>(m_E); \
        ar& boost::serialization::make_nvp(nullptr, tmp);        \
      }                                                          \
      template <typename Archive>                                \
      void load(Archive& ar, const unsigned) {                   \
        std::string tmp;                                         \
        ar& boost::serialization::make_nvp(nullptr, tmp);        \
        m_E = boost::lexical_cast<NAME>(tmp);                    \
      }                                                          \
                                                                 \
     public:                                                     \
      NAME##Clone() = delete;                                    \
      NAME##Clone(const NAME##Clone&) = default;                 \
      NAME##Clone& operator=(NAME##Clone&) = delete;             \
      NAME##Clone(NAME##Clone&&) = default;                      \
      NAME##Clone& operator=(NAME##Clone&&) = default;           \
      explicit NAME##Clone(NAME& e) : m_E(e) {}                  \
    };                                                           \
  }

/// <summary>
/// stream作成マクロ
/// </summary>
/// <param name="NAME">enumの型名</param>
/// <param name="e">enumの値</param>
/// <remarks>
/// ENUM_STREAM_CREATOR で作成したシリアライズでstringに変換するラッパークラスを利用する為のマクロ。
/// </remarks>
#define ENUM_STRING_SERIALIZE(NAME, e) NAME##_string_out::NAME##Clone<NAME>(e)

/// <summary>
/// stream可能enum class作成マクロ
/// </summary>
/// <param name="NS">enumを定義するネームスペース名</param>
/// <param name="NAME">生成するenumの型名</param>
/// <param name="SEQ">生成するenumの値</param>
/// <remarks>
/// ENUM_CLASS_CREATOR , ENUM_STREAM_CREATOR を用いて
/// ストリームで入出力可能なscoped enumを定義するマクロ。
/// ※ VS2010対応
/// </remarks>
#define STREAMABLE_ENUM_CLASS(NAME, SEQ) \
  ENUM_CLASS_CREATOR(NAME, SEQ)          \
  ENUM_STREAM_CREATOR(NAME, SEQ)
