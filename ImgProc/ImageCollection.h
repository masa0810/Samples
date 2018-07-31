#pragma once

#include <Common/OpenCvConfig.h>
#include <Common/SerializeUtility.h>

MSVC_ALL_WARNING_PUSH

#include <cstddef>

#include <ostream>
#include <tuple>
#include <type_traits>
#include <vector>

#include <boost/container_hash/hash.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
#include <boost/operators.hpp>
#include <boost/optional.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>

#include <opencv2/core.hpp>

#include <tbb/cache_aligned_allocator.h>

MSVC_WARNING_POP

/// <summary>
/// 共通ユーティリティ
/// </summary>
namespace commonutility {

/// <summary>
/// 画像設定のパッキング用構造体
/// </summary>
/// @tparam T 画像識別に用いる型
/// @tparam I 画像識別子
/// @tparam V 値タイプ
/// @tparam C チャンネル数
/// @tparam D 画像保存フラグ
template <typename T, T I, typename V, int C = 1, bool D = false>
struct ImageSetting_ {
  //! 識別タイプ
  using Type = T;
  //! 画像タイプ
  using ImageType = std::integral_constant<Type, I>;
  //! 値タイプ
  using ValueType = V;
  //! チャンネルタイプ
  using ChannelType = std::integral_constant<int, C>;
  //! 画像のダンプフラグタイプ
  using DumpImageType = std::integral_constant<bool, D>;
};

/// <summary>
/// 画像設定のパッキング用構造体(intバージョン)
/// </summary>
/// <remarks>自前で識別子(enum等)を用意しない場合はこれを使用</remarks>
/// @tparam I 画像識別子
/// @tparam V 値タイプ
/// @tparam C チャンネル数
/// @tparam D 画像保存フラグ
template <int I, typename V, int C = 1, bool D = false>
using I_ = ImageSetting_<int, I, V, C, D>;


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
  static_assert(!boost::mp11::mp_contains<L, typename I::ImageType>{},
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
template <typename BufferType>
struct ImageBuffer_<BufferType> {
  //! 次のバッファ
  using Tail = std::nullptr_t;

  /// <summary>
  /// cv::Mat取得
  /// </summary>
  /// <returns>cv::Mat</returns>
  cv::Mat Mat() { return {}; }

  /// <summary>
  /// cv::Mat取得
  /// </summary>
  /// <returns>cv::Mat</returns>
  /// <remarks>const用</remarks>
  const cv::Mat Mat() const { return {}; }

 private:
  //! @name シリアライズ用設定
  //@{
  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive&, const unsigned) {}
  //@}
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
  static constexpr auto Channel = typename I::ChannelType{};
  //! 画像のダンプの有無
  static constexpr auto DumpImage = typename I::DumpImageType{};
  //! 要素のバイトサイズ
  static constexpr auto ElemtnSize = sizeof(ValueType);

  //! 画像ステップ
  std::size_t _Step = 0;
  //! 画像バッファサイズ
  cv::Size _Size = {};
  //! 画像バッファ
  BufferType _Buffer = {};

  /// <summary>
  /// cv::Mat取得
  /// </summary>
  /// <returns>cv::Mat</returns>
  cv::Mat Mat() {
    return {this->_Size, cv::Type<ValueType, Channel>(), &this->_Buffer[0], this->_Step};
  }

  /// <summary>
  /// cv::Mat取得
  /// </summary>
  /// <returns>cv::Mat</returns>
  /// <remarks>const用</remarks>
  const cv::Mat Mat() const {
    return {this->_Size, cv::Type<ValueType, Channel>(),
            const_cast<std::uint8_t*>(&this->_Buffer[0]), this->_Step};
  }

 private:
  //! @name シリアライズ用設定
  //@{
  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();
  template <typename Archive>
  void save(Archive& ar, const unsigned) const {
    ar& boost::serialization::make_nvp("Step", this->_Step);
    ar& boost::serialization::make_nvp("Size", this->_Size);
    if (DumpImage) {
      commonutility::Serialize(ar, "Buffer", this->_Buffer);
    }

    ar& boost::serialization::make_nvp("Tail", boost::serialization::base_object<Tail>(*this));
  }
  template <typename Archive>
  void load(Archive& ar, const unsigned) {
    ar& boost::serialization::make_nvp("Step", this->_Step);
    ar& boost::serialization::make_nvp("Size", this->_Size);
    if (DumpImage)
      commonutility::Serialize(ar, "Buffer", this->_Buffer);
    else
      this->_Buffer.assign(this->_Step * this->_Size.height, 0);

    ar& boost::serialization::make_nvp("Tail", boost::serialization::base_object<Tail>(*this));
  }
  //@}
};

/// <summary>
/// 画像バッファの参照
/// </summary>
template <typename BufferType>
struct ImageBufferRef_ {
  //! 画像ステップ
  boost::optional<std::size_t&> StepRef = {};
  //! 画像バッファサイズ
  boost::optional<cv::Size&> SizeRef = {};
  //! 画像バッファ
  boost::optional<BufferType&> BufferRef = {};

  /// <summary>
  /// コンストラクタ
  /// </summary>
  ImageBufferRef_(ImageBuffer_<BufferType>& imgBuf) {}

  /// <summary>
  /// コンストラクタ
  /// </summary>
  template <typename I, typename... Args>
  ImageBufferRef_(ImageBuffer_<BufferType, I, Args...>& imgBuf)
      : StepRef(imgBuf._Step), SizeRef(imgBuf._Size), BufferRef(imgBuf._Buffer) {}

  /// <summary>
  /// コンストラクタ
  /// </summary>
  ImageBufferRef_(const ImageBuffer_<BufferType>& imgBuf) {}

  /// <summary>
  /// コンストラクタ
  /// </summary>
  template <typename I, typename... Args>
  ImageBufferRef_(const ImageBuffer_<BufferType, I, Args...>& imgBuf)
      : StepRef(const_cast<std::size_t&>(imgBuf._Step)),
        SizeRef(const_cast<cv::Size&>(imgBuf._Size)),
        BufferRef(const_cast<BufferType&>(imgBuf._Buffer)) {}

#pragma region デフォルトメソッド定義

  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  ImageBufferRef_() = delete;

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  ImageBufferRef_(const ImageBufferRef_&) = delete;

  /// <summary>
  /// 代入演算子
  /// </summary>
  /// <returns>自分自身</returns>
  ImageBufferRef_& operator=(const ImageBufferRef_&) = delete;

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  ImageBufferRef_(ImageBufferRef_&&) = default;

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  /// <returns>自分自身</returns>
  ImageBufferRef_& operator=(ImageBufferRef_&&) = default;

  /// <summary>
  /// デストラクタ
  /// </summary>
  ~ImageBufferRef_() = default;

#pragma endregion

 private:
  //! @name シリアライズ用設定
  //@{
  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned) {
    if (this->StepRef) ar& boost::serialization::make_nvp("Step", *this->StepRef);
    if (this->SizeRef) ar& boost::serialization::make_nvp("Size", *this->SizeRef);
    if (this->BufferRef) commonutility::Serialize(ar, "Buffer", *this->BufferRef);
  }
  //@}
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
  using Tail = FindImageImpl_<T, typename B::Tail, std::is_same<T, typename B::Tail::ImageType>{}>;
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
  using Tail = FindImageImpl_<T, B, std::is_same<T, typename B::ImageType>{}>;
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
constexpr std::size_t CalcStep(const int width) noexcept {
  // キャッシュライン
  //static const std::size_t CacheLine = tbb::internal::NFS_GetLineSize();
  //static const std::size_t CacheLineM1 = CacheLine - 1;
  static constexpr std::size_t CacheLine = 128;
  static constexpr std::size_t CacheLineM1 = CacheLine - 1;
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
constexpr std::size_t CalcStep(const int width) noexcept {
  // 画像情報
  static constexpr auto channel = ImgBufType::Channel;
  static constexpr auto elemSize = ImgBufType::ElemtnSize;
  static constexpr auto coeff = channel * elemSize;
  // ステップ計算
  return width * coeff;
}

#pragma endregion

#pragma region 再帰実行関数

/// <summary>
/// 次のバッファが無い
/// </summary>
/// <param name="imgBuf">処理する画像バッファ</param>
template <
    typename T, typename F,
    std::enable_if_t<std::is_same<typename T::Tail, std::nullptr_t>{}, std::nullptr_t> = nullptr>
static void Recursive(T&, const F&) {}

/// <summary>
/// 次のバッファがある
/// </summary>
/// <param name="imgBuf">処理する画像バッファ</param>
template <
    typename T, typename F,
    std::enable_if_t<!std::is_same<typename T::Tail, std::nullptr_t>{}, std::nullptr_t> = nullptr>
static void Recursive(T& imgBuf, const F& func) {
  func(imgBuf);
  Recursive<typename T::Tail>(imgBuf, func);
}

/// <summary>
/// 次のバッファが無い
/// </summary>
/// <param name="imgBuf">処理する画像バッファ</param>
template <
    typename T, typename F,
    std::enable_if_t<std::is_same<typename T::Tail, std::nullptr_t>{}, std::nullptr_t> = nullptr>
static void Recursive(const T&, const F&) {}

/// <summary>
/// 次のバッファがある
/// </summary>
/// <param name="imgBuf">処理する画像バッファ</param>
template <
    typename T, typename F,
    std::enable_if_t<!std::is_same<typename T::Tail, std::nullptr_t>{}, std::nullptr_t> = nullptr>
static void Recursive(const T& imgBuf, const F& func) {
  func(imgBuf);
  Recursive<typename T::Tail>(imgBuf, func);
}

#pragma endregion

}  // namespace detail

/// <summary>
/// 画像コレクションクラス
/// </summary>
/// <remarks>std::tuple の実装を参考にした複数の画像のバッファを保持するためのクラス</remarks>
template <template <typename...> class Alloc_, typename I, typename... Args>
class ImageCollectionImpl_
    : detail::DuplicateCheck_<std::tuple<>, I, Args...>,
      detail::ImageBuffer_<std::vector<std::uint8_t, Alloc_<std::uint8_t>>, I, Args...>,
      boost::equality_comparable<ImageCollectionImpl_<Alloc_, I, Args...>,
                                 ImageCollectionImpl_<Alloc_, I, Args...>> {
  //! キャッシュアライメント調整
  static constexpr auto EnableCacheAlign =
      std::is_same<Alloc_<std::uint8_t>, tbb::cache_aligned_allocator<std::uint8_t>>{};
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
  //! 画像バッファ参照タイプ
  using ImageBufferRefType = detail::ImageBufferRef_<BufferType>;

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
  template <>
  struct Impl<> {
    /// <summary>
    /// 初期化
    /// </summary>
    /// <param name="newSize">新しいサイズ</param>
    /// <param name="images">画像コレクションの実体</param>
    static void Init(const cv::Size&, ImageCollectionImpl_&) {}
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
    static void Init(const cv::Size& newSize, ImageCollectionImpl_& images) {
      images.ResizeImpl<ImageBufferType<T>>(newSize, images);
      Impl<Args...>::Init(newSize, images);
    }
  };

#pragma endregion

#pragma region friend宣言

  /// <summary>
  /// ハッシュ関数のフレンド宣言
  /// </summary>
  /// <param name="rhs">[in] ハッシュを計算する対象</param>
  /// <returns>ハッシュ値</returns>
  /// <remarks>boost::hash でハッシュ値を取得出来るようにする。</remarks>
  friend std::size_t hash_value(const ImageCollectionImpl_& rhs) {
    std::size_t hash = 0;
    detail::Recursive<BufferBodyType>(rhs, [&hash](const auto& imgBuf) {
      boost::hash_combine(hash, imgBuf._Step);
      boost::hash_combine(hash, imgBuf._Size);

      using ImgBufType = std::remove_reference_t<std::remove_cv_t<decltype(imgBuf)>>;
      if (ImgBufType::DumpImage) {
        boost::hash_combine(hash, imgBuf._Buffer);
      }
    });
    return hash;
  }

  /// <summary>
  /// 等値比較演算子のフレンド宣言
  /// </summary>
  /// <param name="lhs">[in] 比較対象</param>
  /// <param name="rhs">[in] 比較対象</param>
  /// <returns>比較結果</returns>
  friend bool operator==(const ImageCollectionImpl_& lhs, const ImageCollectionImpl_& rhs) {
    return hash_value(lhs) == hash_value(rhs);
  }

  /// <summary>
  /// ストリーム出力関数のフレンド宣言
  /// </summary>
  /// <param name="os">[in] 出力ストリーム</param>
  /// <param name="rhs">[in] 出力オブジェクト</param>
  /// <returns>ストリーム</returns>
  template <typename charT, typename traits>
  friend std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os,
                                                       const ImageCollectionImpl_& rhs) {
    os << hash_value(rhs);
    return os;
  }

  //! @name シリアライズ用設定
  //@{
  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned) {
    ar& boost::serialization::make_nvp("ImageBuffer",
                                       boost::serialization::base_object<BufferBodyType>(*this));
  }
  //@}

#pragma endregion

  /// <summary>
  /// リサイズ
  /// </summary>
  /// <param name="newSize">新しいサイズ</param>
  template <typename T>
  void ResizeImpl(const cv::Size& newSize, T& imgBuf) {
    // 画像情報
    auto& step = imgBuf._Step;
    auto& size = imgBuf._Size;
    auto& buf = imgBuf._Buffer;

    // ステップ、サイズ、バッファ初期化
    step = detail::CalcStep<T, EnableCacheAlign>(newSize.width);
    size = newSize;
    buf.assign(step * size.height, 0);
  }

 public:
  //! 画像の要素タイプ
  template <ImageType T>
  using ValueType_ = typename ImageBufferType<T>::ValueType;

  //! 画像の要素タイプ
  using ValueType = ValueType_<ImageType{}>;

  /// <summary>
  /// チャンネル数取得
  /// </summary>
  /// <returns>チャンネル数</returns>
  template <ImageType T>
  static constexpr int Channel() {
    return ImageBufferType<T>::Channel;
  }

  /// <summary>
  /// チャンネル数取得
  /// </summary>
  /// <returns>チャンネル数</returns>
  static constexpr int Channel() { return Channel<ImageType{}>(); }

  /// <summary>
  /// 要素サイズ取得
  /// </summary>
  /// <returns>要素サイズ</returns>
  template <ImageType T>
  static constexpr int ElemtnSize() {
    return ImageBufferType<T>::ElemtnSize;
  }

  /// <summary>
  /// 要素サイズ取得
  /// </summary>
  /// <returns>要素サイズ</returns>
  static constexpr int ElemtnSize() { return ElemtnSize<ImageType{}>(); }

  //! 画像要素のcv::Vecタイプ
  template <ImageType T>
  using VecType_ = cv::Vec<ValueType_<T>, Channel<T>()>;

  //! 画像要素のcv::Vecタイプ
  using VecType = VecType_<ImageType{}>;

  /// <summary>
  /// cv::Matのタイプ取得
  /// </summary>
  /// <returns>cv::Matのタイプ</returns>
  template <ImageType T>
  static constexpr int Type() {
    return cv::Type<ValueType_<T>, Channel<T>()>();
  }

  /// <summary>
  /// cv::Matのタイプ取得
  /// </summary>
  /// <returns>cv::Matのタイプ</returns>
  static constexpr int Type() { return Type<ImageType{}>(); }

#pragma region デフォルトメソッド定義

  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  ImageCollectionImpl_() = default;

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  ImageCollectionImpl_(const ImageCollectionImpl_&) = default;

  /// <summary>
  /// 代入演算子
  /// </summary>
  /// <returns>自分自身</returns>
  ImageCollectionImpl_& operator=(const ImageCollectionImpl_&) = default;

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  ImageCollectionImpl_(ImageCollectionImpl_&&) = default;

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  /// <returns>自分自身</returns>
  ImageCollectionImpl_& operator=(ImageCollectionImpl_&&) = default;

  /// <summary>
  /// デストラクタ
  /// </summary>
  virtual ~ImageCollectionImpl_() = default;

#pragma endregion

  /// <summary>
  /// コンストラクタ
  /// </summary>
  /// <param name="newSize">[in] 初期化サイズ</param>
  explicit ImageCollectionImpl_(const cv::Size& newSize) { this->Init(newSize); }

  /// <summary>
  /// 初期化
  /// </summary>
  /// <param name="newSize">[in] 新しいサイズ</param>
  /// <remarks>複数のパラメータリストに対してリサイズを実行する。</remarks>
  template <ImageType... Args>
  void Init(const cv::Size& newSize) {
    Impl<Args...>::Init(newSize, *this);
  }

  /// <summary>
  /// 初期化
  /// </summary>
  /// <param name="newSize">[in] 新しいサイズ</param>
  void Init(const cv::Size& newSize) {
    detail::Recursive<BufferBodyType>(
        *this, [&newSize, this](auto& imgBuf) { this->ResizeImpl(newSize, imgBuf); });
  }

  /// <summary>
  /// 初期化
  /// </summary>
  /// <param name="newWidth">[in] 新しい幅</param>
  /// <param name="newHeight">[in] 新しい高さ</param>
  /// <remarks>複数のパラメータリストに対してリサイズを実行する。</remarks>
  template <ImageType... Args>
  void Init(const int newWidth, const int newHeight) {
    this->Init<Args...>({newWidth, newHeight});
  }

  /// <summary>
  /// 初期化
  /// </summary>
  /// <param name="newWidth">[in] 新しいサイズ</param>
  /// <param name="newHeight">[in] 新しいサイズ</param>
  void Init(const int newWidth, const int newHeight) { this->Init({newWidth, newHeight}); }

  /// <summary>
  /// クリア
  /// </summary>
  template <ImageType... Args>
  void Clear() {
    this->Init<Args...>({0, 0});
  }

  /// <summary>
  /// クリア
  /// </summary>
  void Clear() { this->Init({0, 0}); }

  /// <summary>
  /// RAWデータ取得
  /// </summary>
  /// <returns>RAWデータ</returns>
  template <ImageType T>
  ImageBufferRefType Raw() {
    ImageBufferType<T>& imgBuf = *this;
    return imgBuf;
  }

  /// <summary>
  /// RAWデータ取得
  /// </summary>
  /// <returns>RAWデータ</returns>
  /// <remarks>最初のデータへのアクセス</remarks>
  ImageBufferRefType Raw() {
    ImageBufferType<ImageType{}>& imgBuf = *this;
    return imgBuf;
  }

  /// <summary>
  /// RAWデータ取得
  /// </summary>
  /// <returns>RAWデータ</returns>
  /// <remarks>const用</remarks>
  template <ImageType T>
  const ImageBufferRefType Raw() const {
    const ImageBufferType<T>& imgBuf = *this;
    return imgBuf;
  }

  /// <summary>
  /// RAWデータ取得
  /// </summary>
  /// <returns>RAWデータ</returns>
  /// <remarks>
  /// <para>const用</para>
  /// <para>最初のデータへのアクセス</para>
  /// </remarks>
  const ImageBufferRefType Raw() const {
    const ImageBufferType<ImageType{}>& imgBuf = *this;
    return imgBuf;
  }

  /// <summary>
  /// cv::Mat取得
  /// </summary>
  /// <returns>cv::Mat</returns>
  template <ImageType T>
  cv::Mat Mat() {
    using ImgBufType = ImageBufferType<T>;
    ImgBufType& imgBuf = *this;
    return imgBuf.Mat();
  }

  /// <summary>
  /// cv::Mat取得
  /// </summary>
  /// <returns>cv::Mat</returns>
  /// <remarks>最初のデータへのアクセス</remarks>
  cv::Mat Mat() { return this->Mat<ImageType{}>(); }

  /// <summary>
  /// cv::Mat取得
  /// </summary>
  /// <returns>cv::Mat</returns>
  /// <remarks>const用</remarks>
  template <ImageType T>
  const cv::Mat Mat() const {
    using ImgBufType = ImageBufferType<T>;
    const ImgBufType& imgBuf = *this;
    return imgBuf.Mat();
  }

  /// <summary>
  /// cv::Mat取得
  /// </summary>
  /// <returns>cv::Mat</returns>
  /// <remarks>
  /// <para>const用</para>
  /// <para>最初のデータへのアクセス</para>
  /// </remarks>
  const cv::Mat Mat() const { return this->Mat<ImageType{}>(); }

  /// <summary>
  /// cv::Mat取得
  /// </summary>
  /// <param name="roi">[in] ROI</param>
  /// <returns>cv::Mat</returns>
  /// <remarks>const用</remarks>
  template <ImageType T>
  const cv::Mat Mat(const cv::Rect& roi) const {
    return this->Mat<T>()(roi);
  }

  /// <summary>
  /// cv::Mat取得
  /// </summary>
  /// <param name="roi">[in] ROI</param>
  /// <returns>cv::Mat</returns>
  /// <remarks>
  /// <para>const用</para>
  /// <para>最初のデータへのアクセス</para>
  /// </remarks>
  const cv::Mat Mat(const cv::Rect& roi) const { return this->Mat()(roi); }

  /// <summary>
  /// ステップ取得
  /// </summary>
  /// <returns>ステップ</returns>
  template <ImageType T>
  std::size_t GetStep() const {
    return ImageBufferType<T>::_Step;
  }

  /// <summary>
  /// ステップ取得
  /// </summary>
  /// <returns>ステップ</returns>
  /// <remarks>最初のデータへのアクセス</remarks>
  std::size_t GetStep() const {
    return this->GetStep<ImageType{}>();
    ;
  }

  /// <summary>
  /// サイズ取得
  /// </summary>
  /// <returns>サイズ</returns>
  template <ImageType T>
  const cv::Size& GetSize() const {
    return ImageBufferType<T>::_Size;
  }

  /// <summary>
  /// サイズ取得
  /// </summary>
  /// <returns>サイズ</returns>
  /// <remarks>最初のデータへのアクセス</remarks>
  const cv::Size& GetSize() const { return this->GetSize<ImageType{}>(); }

  /// <summary>
  /// 空チェック
  /// </summary>
  /// <returns>空かどうか</returns>
  template <ImageType T>
  bool Empty() const {
    const auto& buf = ImageBufferType<T>::_Buffer;
    return buf.empty();
  }

  /// <summary>
  /// 空チェック
  /// </summary>
  /// <returns>空かどうか</returns>
  /// <remarks>最初のデータへのアクセス</remarks>
  bool Empty() const { return this->Empty<ImageType{}>(); }
};

/// <summary>
/// 画像コレクション
/// </summary>
template <typename... Args>
using ImageCollection_ = ImageCollectionImpl_<std::allocator, Args...>;

/// <summary>
/// 画像コレクション
/// </summary>
/// <remarks>アライメント調整済み</remarks>
template <typename... Args>
using AlignedImageCollection_ = ImageCollectionImpl_<tbb::cache_aligned_allocator, Args...>;

/// <summary>
/// 画像バッファの詳細定義
/// </summary>
template <template <typename...> class Alloc_, typename V, int C = 1, bool D = false>
using ImageBufferImpl_ = ImageCollectionImpl_<Alloc_, I_<0, V, C, D>>;

/// <summary>
/// 画像バッファ
/// </summary>
template <typename V, int C = 1, bool D = false>
using ImageBuffer_ = ImageBufferImpl_<std::allocator, V, C, D>;

/// <summary>
/// 画像バッファ
/// </summary>
template <typename V, int C = 1, bool D = false>
using AlignedImageBuffer_ = ImageBufferImpl_<tbb::cache_aligned_allocator, V, C, D>;

}  // namespace commonutility
