#pragma once

#include <Common/OpenCvConfig.h>

MSVC_ALL_WARNING_PUSH

#include <cstddef>

#include <tuple>
#include <type_traits>
#include <vector>

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

#include <opencv2/core.hpp>

#include <tbb/cache_aligned_allocator.h>

MSVC_WARNING_POP

namespace bmp11 {

/// <summary>
/// 画像設定のパッキング用構造体
/// </summary>
template <typename T, T I, typename V, int C>
struct ImageSetting_ {
  using Type = T;
  using ImageType = std::integral_constant<Type, I>;
  using ValueType = V;
  using ChannelType = std::integral_constant<int, C>;
};

/// <summary>
/// 詳細
/// </summary>
namespace detail {

#pragma region 重複チェック

/// <summary>
/// 宣言
/// </summary>
template <typename L, typename... Args>
struct DuplicateCheck_;

/// <summary>
/// 特殊化
/// </summary>
/// <remarks>パラメータリストの1つ</remarks>
template <typename L, typename I>
struct DuplicateCheck_<L, I> {
  //! 画像タイプのリスト
  using ImageTypeList = boost::mp11::mp_push_back<L, typename I::ImageType>;
};

/// <summary>
/// 実装の特殊化
/// </summary>
/// <remarks>パラメータリストが複数</remarks>
template <typename L, typename I, typename... Args>
struct DuplicateCheck_<L, I, Args...>
    : DuplicateCheck_<boost::mp11::mp_push_back<L, typename I::ImageType>, Args...> {
  //! 重複エラー
  static_assert(!boost::mp11::mp_contains<L, typename I::ImageType>::value,
                "Duplicate declaration of the image type is not allowed.");
  //! 画像タイプのリスト
  using ImageTypeList =
      typename DuplicateCheck_<boost::mp11::mp_push_back<L, typename I::ImageType>,
                               Args...>::ImageTypeList;
};

#pragma endregion

#pragma region 画像バッファ

/// <summary>
/// 宣言
/// </summary>
template <typename BufferType, typename... Args>
struct ImageBuffer_;

/// <summary>
/// 特殊化
/// </summary>
/// <remarks>パラメータリストの1つ</remarks>
template <typename BufferType, typename I>
struct ImageBuffer_<BufferType, I> {
  //! 画像タイプ
  using ImageType = typename I::ImageType;
  //! 画像の要素タイプ
  using ValueType = typename I::ValueType;
  //! チャンネル
  static constexpr auto Channel = typename I::ChannelType::value;
  //! 要素のバイトサイズ
  static constexpr auto ElemtnSize = sizeof(ValueType);

  //! 画像ステップ
  std::size_t _Step = cv::Mat::AUTO_STEP;
  //! 画像バッファサイズ
  cv::Size _Size = {};
  //! 画像バッファ
  BufferType _Buffer = {};
};

/// <summary>
/// 実装の特殊化
/// </summary>
/// <remarks>パラメータリストが複数</remarks>
template <typename BufferType, typename I, typename... Args>
struct ImageBuffer_<BufferType, I, Args...> : ImageBuffer_<BufferType, Args...> {
  //! 次のバッファ
  using Tail = ImageBuffer_<BufferType, Args...>;
  //! 画像タイプ
  using ImageType = typename I::ImageType;
  //! 画像の要素タイプ
  using ValueType = typename I::ValueType;
  //! チャンネル
  static constexpr auto Channel = typename I::ChannelType::value;
  //! 要素のバイトサイズ
  static constexpr auto ElemtnSize = sizeof(ValueType);

  //! 画像ステップ
  std::size_t _Step = cv::Mat::AUTO_STEP;
  //! 画像バッファサイズ
  cv::Size _Size = {};
  //! 画像バッファ
  BufferType _Buffer = {};
};

#pragma endregion

#pragma region 画像バッファ探索

#pragma region 詳細

/// <summary>
/// 宣言
/// </summary>
template <typename T, typename B, bool found>
struct FindImageImpl_;

/// <summary>
/// 特殊化
/// </summary>
/// <remarks>見つかった</remarks>
template <typename T, typename B>
struct FindImageImpl_<T, B, true> {
  //! 見つけた画像バッファタイプ
  using ImageBufferType = B;
};

/// <summary>
/// 特殊化
/// </summary>
/// <remarks>見つかってない</remarks>
template <typename T, typename B>
struct FindImageImpl_<T, B, false>
    : FindImageImpl_<T, typename B::Tail, std::is_same<T, typename B::Tail::ImageType>::value> {
  //! 次の画像バッファ
  using Tail =
      FindImageImpl_<T, typename B::Tail, std::is_same<T, typename B::Tail::ImageType>::value>;
  //! 最終的に見つけた画像バッファタイプ
  using ImageBufferType = typename Tail::ImageBufferType;
};

#pragma endregion

/// <summary>
/// 画像バッファ探索
/// </summary>
template <typename T, typename B>
struct FindImage_ : FindImageImpl_<T, B, std::is_same<T, typename B::ImageType>::value> {
  //! 次の画像バッファ
  using Tail = FindImageImpl_<T, B, std::is_same<T, typename B::ImageType>::value>;
  //! 最終的に見つけた画像バッファタイプ
  using ImageBufferType = typename Tail::ImageBufferType;
};

#pragma endregion

#pragma region 画像ステップ計算

/// <summary>
/// 画像ステップ計算
/// </summary>
/// <param name="width">画像幅</param>
/// <returns>画像ステップ</returns>
/// <remarks>キャッシュアライメント有効</remarks>
template <typename ImgBufType, bool enableCacheAlign,
          std::enable_if_t<enableCacheAlign, std::nullptr_t> = nullptr>
std::size_t CalcStep(const int width) {
  // キャッシュライン
  static const std::size_t CacheLine = tbb::internal::NFS_GetLineSize();
  static const std::size_t CacheLineM1 = CacheLine - 1;
  // 画像情報
  static constexpr auto channel = ImgBufType::Channel;
  static constexpr auto elemSize = ImgBufType::ElemtnSize;
  static constexpr auto coeff = channel * elemSize;
  // ステップ計算
  return (((width * coeff) + CacheLineM1) / CacheLine) * CacheLine;
}

/// <summary>
/// 画像ステップ計算
/// </summary>
/// <param name="width">画像幅</param>
/// <returns>画像ステップ</returns>
/// <remarks>キャッシュアライメント無効</remarks>
template <typename ImgBufType, bool enableCacheAlign,
          std::enable_if_t<!enableCacheAlign, std::nullptr_t> = nullptr>
std::size_t CalcStep(const int width) {
  // 画像情報
  static constexpr auto channel = ImgBufType::Channel;
  static constexpr auto elemSize = ImgBufType::ElemtnSize;
  static constexpr auto coeff = channel * elemSize;
  // ステップ計算
  return width * coeff;
}

#pragma endregion

}  // namespace detail

/// <summary>
/// 画像コレクションクラス
/// </summary>
template <template <typename...> class Alloc_, bool EnableCacheAlign, typename I, typename... Args>
class _ImageCollection_
    : public detail::DuplicateCheck_<std::tuple<>, I, Args...>,
      detail::ImageBuffer_<std::vector<std::uint8_t, Alloc_<std::uint8_t>>, I, Args...> {
  //! バッファタイプ
  using BufferType = std::vector<std::uint8_t, Alloc_<std::uint8_t>>;
  //! バッファ本体タイプ
  using BufferBodyType = detail::ImageBuffer_<BufferType, I, Args...>;
  //! 画像種別enumタイプ
  using ImageType = typename I::Type;
  //! 画像バッファタイプ
  template <ImageType T>
  using ImageBufferType = typename detail::FindImage_<std::integral_constant<ImageType, T>,
                                                      BufferBodyType>::ImageBufferType;

#pragma region 複数画像初期化

  /// <summary>
  /// 宣言
  /// </summary>
  /// <remarks>テンプレートパラメータの特殊化のため</remarks>
  template <ImageType... Args>
  struct Impl;

  /// <summary>
  /// 実装の特殊化
  /// </summary>
  /// <remarks>パラメータリストの1つ</remarks>
  template <ImageType T>
  struct Impl<T> {
    /// <summary>
    /// 初期化
    /// </summary>
    /// <param name="newSize">新しいサイズ</param>
    /// <param name="images">画像コレクションの実体</param>
    static void Init(const cv::Size& newSize, _ImageCollection_& images) {
      images.Resize<T>(newSize);
    }
  };

  /// <summary>
  /// 実装の特殊化
  /// </summary>
  /// <remarks>パラメータリストが複数</remarks>
  template <ImageType T, ImageType... Args>
  struct Impl<T, Args...> {
    /// <summary>
    /// 初期化
    /// </summary>
    /// <param name="newSize">新しいサイズ</param>
    /// <param name="images">画像コレクションの実体</param>
    static void Init(const cv::Size& newSize, _ImageCollection_& images) {
      images.Resize<T>(newSize);
      Impl<Args...>::Init(newSize, images);
    }
  };

#pragma endregion

 public:
  //! 画像の要素タイプ
  template <ImageType T>
  using ValueType = typename ImageBufferType<T>::ValueType;

  /// <summary>
  /// チャンネル数取得
  /// </summary>
  /// <returns>チャンネル数</returns>
  template <ImageType T>
  static constexpr int Channel() {
    return ImageBufferType<T>::Channel;
  }

  /// <summary>
  /// 要素サイズ取得
  /// </summary>
  /// <returns>要素サイズ</returns>
  template <ImageType T>
  static constexpr int ElemtnSize() {
    return ImageBufferType<T>::ElemtnSize;
  }

  //! 画像要素のcv::Vecタイプ
  template <ImageType T>
  using VecType = cv::Vec<ValueType<T>, Channel<T>()>;

  /// <summary>
  /// cv::Matのタイプ取得
  /// </summary>
  /// <returns>cv::Matのタイプ</returns>
  template <ImageType T>
  static constexpr int Type() {
    return cv::Type<ValueType<T>, Channel<T>()>();
  }

#pragma region デフォルトメソッド定義

  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  _ImageCollection_() = default;

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  _ImageCollection_(const _ImageCollection_&) = default;

  /// <summary>
  /// 代入演算子
  /// </summary>
  /// <returns>自分自身</returns>
  _ImageCollection_& operator=(const _ImageCollection_&) = default;

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  _ImageCollection_(_ImageCollection_&&) = default;

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  /// <returns>自分自身</returns>
  _ImageCollection_& operator=(_ImageCollection_&&) = default;

  /// <summary>
  /// デストラクタ
  /// </summary>
  virtual ~_ImageCollection_() = default;

#pragma endregion

  /// <summary>
  /// 初期化
  /// </summary>
  /// <param name="newSize">新しいサイズ</param>
  /// <remarks>複数のパラメータリストに対してリサイズを実行する。</remarks>
  template <ImageType... Args>
  void Init(const cv::Size& newSize) {
    Impl<Args...>::Init(newSize, *this);
  }

  /// <summary>
  /// リサイズ
  /// </summary>
  /// <param name="newSize">新しいサイズ</param>
  template <ImageType T>
  void Resize(const cv::Size& newSize) {
    // キャッシュライン
    static const std::size_t CacheLine = tbb::internal::NFS_GetLineSize();
    static const std::size_t CacheLineM1 = CacheLine - 1;

    // 画像情報
    using ImgBufType = ImageBufferType<T>;
    static constexpr auto channel = ImgBufType::Channel;
    static constexpr auto elemSize = ImgBufType::ElemtnSize;
    static constexpr auto coeff = channel * elemSize;
    ImgBufType& imgBuf = *this;
    auto& step = imgBuf._Step;
    auto& size = imgBuf._Size;
    auto& buf = imgBuf._Buffer;

    // ステップ、サイズ、バッファ初期化
    step = detail::CalcStep<ImgBufType, EnableCacheAlign>(newSize.width);
    size = newSize;
    buf.assign(step * size.height, std::uint8_t(0));
  }

  /// <summary>
  /// クリア
  /// </summary>
  template <ImageType... Args>
  void Clear() {
    this->Init<Args...>({0, 0});
  }

  /// <summary>
  /// cv::Mat取得
  /// </summary>
  template <ImageType T>
  cv::Mat Mat() {
    using ImgBufType = ImageBufferType<T>;
    using ValType = typename ImgBufType::ValueType;
    static constexpr auto channel = ImgBufType::Channel;
    static constexpr auto type = cv::Type<ValType, channel>();
    ImgBufType& imgBuf = *this;
    const auto& step = imgBuf._Step;
    const auto& size = imgBuf._Size;
    auto& buf = imgBuf._Buffer;
    return {size, type, &buf[0], step};
  }

  /// <summary>
  /// cv::Mat取得
  /// </summary>
  /// <remarks>const用</remarks>
  template <ImageType T>
  const cv::Mat Mat() const {
    using ImgBufType = ImageBufferType<T>;
    using ValType = typename ImgBufType::ValueType;
    static constexpr auto channel = ImgBufType::Channel;
    static constexpr auto type = cv::Type<ValType, channel>();
    const ImgBufType& imgBuf = *this;
    const auto& step = imgBuf._Step;
    const auto& size = imgBuf._Size;
    const auto& buf = imgBuf._Buffer;
    return {size, type, const_cast<std::uint8_t*>(&buf[0]), step};
  }

  /// <summary>
  /// ステップ取得
  /// </summary>
  /// <returns>ステップ</returns>
  /// <remarks>const用</remarks>
  template <ImageType T>
  std::size_t GetStep() const {
    return ImageBufferType<T>::_Step;
  }

  /// <summary>
  /// サイズ取得
  /// </summary>
  /// <returns>サイズ</returns>
  /// <remarks>const用</remarks>
  template <ImageType T>
  const cv::Size& GetSize() const {
    return ImageBufferType<T>::_Size;
  }

  /// <summary>
  /// 空チェック
  /// </summary>
  template <ImageType T>
  bool Empty() const {
    const auto& buf = ImageBufferType<T>::_Buffer;
    return buf.empty();
  }
};

/// <summary>
/// 画像コレクション
/// </summary>
/// <remarks>アロケータ指定</remarks>
template <typename... Args>
using ImageCollection_ = _ImageCollection_<tbb::cache_aligned_allocator, true, Args...>;

}  // namespace bmp11
