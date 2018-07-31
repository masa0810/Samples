#pragma once

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <cstddef>

#include <functional>
#include <tuple>

// 警告抑制解除
MSVC_WARNING_POP

/// <summary>
/// 共通ユーティリティ
/// </summary>
namespace commonutility {

/// <summary>
/// コピーしないメンバー定義のためのクラス
/// </summary>
/// <remarks>
/// <para>std::mutex などコピーやムーブが禁止されているクラスをメンバーとして持った場合、</para>
/// <para>コピーコンストラクタなどのコンパイラ定義のメソッドを自分で定義しなくてはならない。</para>
/// <para>その定義を簡略化するためのクラス。</para>
/// </remarks>
/// @tparam Types メンバータイプ
template <typename... Types>
class NonCopyMembers_ {
  //! タプルタイプ
  using TupleType = std::tuple<Types...>;
  //! メンバ
  mutable TupleType m_members = {};
  //! 初期化用関数オブジェクト
  std::function<void(TupleType&)> m_initFunc = {};

 public:
  //! メンバ数
  static constexpr auto NumOfMembers = std::tuple_size<TupleType>::value;
  //! メンバタイプ
  //!@tparam I メンバインデックス
  template <std::size_t I>
  using NcMemberType_ = std::tuple_element_t<I, TupleType>;
  //! メンバタイプ(インデックス0の要素)
  using NcMemberType = std::tuple_element_t<0, TupleType>;

#pragma region デフォルトメソッド定義

  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  NonCopyMembers_() = default;

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  /// <param name="rhs">コピー元</param>
  NonCopyMembers_(const NonCopyMembers_& rhs) : m_initFunc(rhs.m_initFunc) {
    if (m_initFunc) m_initFunc(m_members);
  }

  /// <summary>
  /// 代入演算子
  /// </summary>
  NonCopyMembers_& operator=(const NonCopyMembers_&) { return *this; }

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  /// <param name="rhs">ムーブ元</param>
  NonCopyMembers_(NonCopyMembers_&& rhs) : m_initFunc(std::move(rhs.m_initFunc)) {
    if (m_initFunc) m_initFunc(m_members);
  }

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  NonCopyMembers_& operator=(NonCopyMembers_&&) { return *this; }

  /// <summary>
  /// デストラクタ
  /// </summary>
  ~NonCopyMembers_() = default;

#pragma endregion

  /// <summary>
  /// コンストラクタ
  /// </summary>
  /// <param name="initFunc">初期化用関数オブジェクト</param>
  /// <remarks>メンバの初期化をカスタマイズ出来るコンストラクタ</remarks>
  /// @tparam F 初期化用関数オブジェクトタイプ
  template <typename F>
  NonCopyMembers_(const F& initFunc) : m_initFunc(initFunc) {}

  /// <summary>
  /// メンバアクセス
  /// </summary>
  /// <returns>メンバへの参照</returns>
  /// @tparam I メンバインデックス
  template <std::size_t I>
  NcMemberType_<I>& NcMember() noexcept {
    return std::get<I>(m_members);
  }

  /// <summary>
  /// メンバアクセス
  /// </summary>
  /// <returns>メンバへの参照</returns>
  /// <remarks>インデックス0の要素専用</remarks>
  NcMemberType& NcMember() noexcept { return std::get<0>(m_members); }

  /// <summary>
  /// メンバアクセス
  /// </summary>
  /// <returns>メンバへの参照</returns>
  /// <remarks>const用</remarks>
  /// @tparam I メンバインデックス
  template <std::size_t I>
  const NcMemberType_<I>& NcMember() const noexcept {
    return std::get<I>(m_members);
  }

  /// <summary>
  /// メンバアクセス
  /// </summary>
  /// <returns>メンバへの参照</returns>
  /// <remarks>
  /// <para>インデックス0の要素専用</para>
  /// <para>const用</para>
  /// </remarks>
  const NcMemberType& NcMember() const noexcept { return std::get<0>(m_members); }

  /// <summary>
  /// メンバアクセス
  /// </summary>
  /// <returns>メンバへの参照</returns>
  /// <remarks>const用</remarks>
  /// @tparam I メンバインデックス
  template <std::size_t I>
  NcMemberType_<I>& NcMemberMutable() const noexcept {
    return std::get<I>(m_members);
  }

  /// <summary>
  /// メンバアクセス
  /// </summary>
  /// <returns>メンバへの参照</returns>
  /// <remarks>
  /// <para>インデックス0の要素専用</para>
  /// <para>mutable アクセス用</para>
  /// </remarks>
  NcMemberType& NcMemberMutable() const noexcept { return std::get<0>(m_members); }

  /// <summary>
  /// メンバアクセス
  /// </summary>
  /// <returns>メンバへの参照</returns>
  /// @tparam T メンバタイプ
  template <typename T>
  T& NcMember() noexcept {
    return std::get<T>(m_members);
  }

  /// <summary>
  /// メンバアクセス
  /// </summary>
  /// <returns>メンバへの参照</returns>
  /// <remarks>const用</remarks>
  /// @tparam T メンバタイプ
  template <typename T>
  const T& NcMember() const noexcept {
    return std::get<T>(m_members);
  }

  /// <summary>
  /// メンバアクセス
  /// </summary>
  /// <returns>メンバへの参照</returns>
  /// <remarks>mutable アクセス用</remarks>
  /// @tparam T メンバタイプ
  template <typename T>
  T& NcMemberMutable() const noexcept {
    return std::get<T>(m_members);
  }
};

}  // namespace commonutility
