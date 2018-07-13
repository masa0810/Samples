#pragma once

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <algorithm>
#include <forward_list>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_member.hpp>

#include <tbb/cache_aligned_allocator.h>
#include <tbb/scalable_allocator.h>

// 警告抑制解除
MSVC_WARNING_POP

namespace commonutility {

/// <summary>
/// 動的バッファ
/// </summary>
/// <remarks>
/// <para>バッファ用なので、バッファの中身はシリアライズされない。</para>
/// </remarks>
template <typename T, template <typename> class BufAlloc_ = tbb::cache_aligned_allocator,
          template <typename> class CacheAlloc_ = tbb::scalable_allocator>
class DynamicBuffer {
 public:
  //! バッファータイプ
  using BufferType = T;

 private:
  //! バッファー参照タイプ
  using BufferRefType = std::reference_wrapper<BufferType>;
  //! バッファーの実体配列
  using BufferListType = std::forward_list<BufferType, CacheAlloc_<BufferType>>;
  //! バッファーの参照配列
  using BufferRefsType = std::vector<BufferRefType, CacheAlloc_<BufferRefType>>;

  //! バッファの実体
#if !defined(__GNUC__) || (__GNUC__ > 5)
  BufferListType m_buffers = {};
#else
  BufferListType m_buffers = BufferListType();
#endif
  //! 待機バッファ
  BufferRefsType m_storeBuffers = {};

  //! ミューテックス
  std::mutex m_mutex = {};

  //! @name シリアライズ用設定
  //@{
  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();
  template <typename Archive>
  void save(Archive& ar, const unsigned) const {
    ar << boost::serialization::make_nvp("BufferSize", m_bufSize);

    auto numOfBuffers = m_storeBuffers.size();
    ar << boost::serialization::make_nvp("NumOfBuffers", numOfBuffers);
  }
  template <typename Archive>
  void load(Archive& ar, const unsigned) {
    ar >> boost::serialization::make_nvp("BufferSize", m_bufSize);

    size_t numOfBuffers = 0;
    ar >> boost::serialization::make_nvp("NumOfBuffers", numOfBuffers);

    m_buffers.clear();
    m_storeBuffers.reserve(numOfBuffers);
    m_storeBuffers.clear();
    for (auto i = 0; i < numOfBuffers; ++i) {
      m_buffers.emplace_front();
      m_storeBuffers.emplace_back(m_buffers.front());
    }
  }
  //@}

 public:
#pragma region デフォルトメソッド定義

  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  DynamicBuffer() = default;

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  /// <param name="rhs">コピー元</param>
  DynamicBuffer(const DynamicBuffer& rhs) {
    m_buffers.clear();
    m_storeBuffers.reserve(rhs.m_storeBuffers.size());
    m_storeBuffers.clear();
    for (const auto& refBuf : rhs.m_storeBuffers) {
      m_buffers.emplace_front(refBuf.get());
      m_storeBuffers.emplace_back(m_buffers.front());
    }
  }

  /// <summary>
  /// スワップ
  /// </summary>
  /// <param name="rhs">スワップ対象</param>
  void swap(DynamicBuffer& rhs) noexcept {
    std::swap(m_buffers, rhs.m_buffers);
    std::swap(m_storeBuffers, rhs.m_storeBuffers);
  }

  /// <summary>
  /// 代入演算子
  /// </summary>
  /// <param name="rhs">コピー元</param>
  DynamicBuffer& operator=(const DynamicBuffer& rhs) {
    auto tmp = rhs;
    this->swap(tmp);
    return *this;
  }

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  /// <param name="rhs">ムーブ元</param>
  DynamicBuffer(DynamicBuffer&& rhs)
      : m_buffers(std::move(rhs.m_buffers)), m_storeBuffers(std::move(rhs.m_storeBuffers)) {}

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  /// <param name="rhs">ムーブ元</param>
  DynamicBuffer& operator=(DynamicBuffer&& rhs) {
    m_buffers = std::move(rhs.m_buffers);
    m_storeBuffers = std::move(rhs.m_storeBuffers);
    return *this;
  }

  /// <summary>
  /// デストラクタ
  /// </summary>
  virtual ~DynamicBuffer() = default;

#pragma endregion

  // バッファ取得
  BufferType& Get() {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_storeBuffers.empty()) {
      m_buffers.emplace_front();
      m_storeBuffers.emplace_back(m_buffers.front());
    }

    auto& buffer = m_storeBuffers.back().get();
    m_storeBuffers.pop_back();

    return buffer;
  }

  // バッファ解放
  void Release(BufferType& buf) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_storeBuffers.emplace_back(buf);
  }

  // クリア
  void Clear() {
    m_storeBuffers.clear();
    m_buffers.clear();
  }

  // サイズ取得
  int GetSize() const { return m_buffers.size(); }

  class Scoped {
    DynamicBuffer& m_dynamincBuf;

   public:
    BufferType& Buf;

#pragma region デフォルトメソッド定義

    /// <summary>
    /// デフォルトコンストラクタ
    /// </summary>
    Scoped() = delete;

    /// <summary>
    /// コピーコンストラクタ
    /// </summary>
    Scoped(const Scoped&) = delete;

    /// <summary>
    /// 代入演算子
    /// </summary>
    Scoped& operator=(const Scoped&) = delete;

    /// <summary>
    /// ムーブコンストラクタ
    /// </summary>
    Scoped(Scoped&&) = default;

    /// <summary>
    /// ムーブ代入演算子
    /// </summary>
    Scoped& operator=(Scoped&&) = default;

    /// <summary>
    /// デストラクタ
    /// </summary>
    ~Scoped() { m_dynamincBuf.Release(Buf); }

#pragma endregion

    /// <summary>
    /// コンストラクタ
    /// </summary>
    explicit Scoped(DynamicBuffer& dynamicBuf)
        : m_dynamincBuf(dynamicBuf), Buf(m_dynamincBuf.Get()) {}
  };
};

/// <summary>
/// 動的バッファ
/// </summary>
/// <remarks>
/// <para>バッファ用なので、バッファの中身はシリアライズされない。</para>
/// <para>配列バッファバージョン。</para>
/// </remarks>
template <typename T, template <typename> class BufAlloc_ = tbb::cache_aligned_allocator,
          template <typename> class CacheAlloc_ = tbb::scalable_allocator>
class DynamicVector {
 public:
  //! バッファータイプ
  using BufferType = std::vector<T, BufAlloc_<T>>;

 private:
  //! バッファー参照タイプ
  using BufferRefType = std::reference_wrapper<BufferType>;
  //! バッファーの実体配列
  using BufferListType = std::forward_list<BufferType, CacheAlloc_<BufferType>>;
  //! バッファーの参照配列
  using BufferRefsType = std::vector<BufferRefType, CacheAlloc_<BufferRefType>>;

  //! バッファサイズ
  int m_bufSize = 0;
  //! バッファの初期値
  T m_initValue = {};
  //! バッファの実体
#if !defined(__GNUC__) || (__GNUC__ > 5)
  BufferListType m_buffers = {};
#else
  BufferListType m_buffers = BufferListType();
#endif
  //! 待機バッファ
  BufferRefsType m_storeBuffers = {};

  //! ミューテックス
  std::mutex m_mutex = {};

  //! @name シリアライズ用設定
  //@{
  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();
  template <typename Archive>
  void save(Archive& ar, const unsigned) const {
    ar << boost::serialization::make_nvp("BufferSize", m_bufSize);
    ar << boost::serialization::make_nvp("InitValue", m_initValue);

    auto numOfBuffers = m_storeBuffers.size();
    ar << boost::serialization::make_nvp("NumOfBuffers", numOfBuffers);
  }
  template <typename Archive>
  void load(Archive& ar, const unsigned) {
    ar >> boost::serialization::make_nvp("BufferSize", m_bufSize);
    ar >> boost::serialization::make_nvp("InitValue", m_initValue);

    size_t numOfBuffers = 0;
    ar >> boost::serialization::make_nvp("NumOfBuffers", numOfBuffers);

    m_buffers.clear();
    m_storeBuffers.reserve(numOfBuffers);
    m_storeBuffers.clear();
    for (auto i = 0; i < numOfBuffers; ++i) {
      m_buffers.emplace_front(m_bufSize, m_initValue);
      m_storeBuffers.emplace_back(m_buffers.front());
    }
  }
  //@}

 public:
#pragma region デフォルトメソッド定義

  /// <summary>
  /// デフォルトコンストラクタ
  /// </summary>
  DynamicVector() = default;

  /// <summary>
  /// コピーコンストラクタ
  /// </summary>
  /// <param name="rhs">コピー元</param>
  DynamicVector(const DynamicVector& rhs) : m_bufSize(rhs.m_bufSize), m_initValue(rhs.m_initValue) {
    m_buffers.clear();
    m_storeBuffers.reserve(rhs.m_storeBuffers.size());
    m_storeBuffers.clear();
    for (const auto& refBuf : rhs.m_storeBuffers) {
      m_buffers.emplace_front(refBuf.get());
      m_storeBuffers.emplace_back(m_buffers.front());
    }
  }

  /// <summary>
  /// スワップ
  /// </summary>
  /// <param name="rhs">スワップ対象</param>
  void swap(DynamicVector& rhs) noexcept {
    std::swap(m_bufSize, rhs.m_bufSize);
    std::swap(m_initValue, rhs.m_initValue);
    std::swap(m_buffers, rhs.m_buffers);
    std::swap(m_storeBuffers, rhs.m_storeBuffers);
  }

  /// <summary>
  /// 代入演算子
  /// </summary>
  /// <param name="rhs">コピー元</param>
  DynamicVector& operator=(const DynamicVector& rhs) {
    auto tmp = rhs;
    this->swap(tmp);
    return *this;
  }

  /// <summary>
  /// ムーブコンストラクタ
  /// </summary>
  /// <param name="rhs">ムーブ元</param>
  DynamicVector(DynamicVector&& rhs)
      : m_bufSize(std::move(rhs.m_bufSize)),
        m_initValue(std::move(rhs.m_initValue)),
        m_buffers(std::move(rhs.m_buffers)),
        m_storeBuffers(std::move(rhs.m_storeBuffers)) {}

  /// <summary>
  /// ムーブ代入演算子
  /// </summary>
  /// <param name="rhs">ムーブ元</param>
  DynamicVector& operator=(DynamicVector&& rhs) {
    m_bufSize = std::move(rhs.m_bufSize);
    m_initValue = std::move(rhs.m_initValue);
    m_buffers = std::move(rhs.m_buffers);
    m_storeBuffers = std::move(rhs.m_storeBuffers);
    return *this;
  }

  /// <summary>
  /// デストラクタ
  /// </summary>
  virtual ~DynamicVector() = default;

#pragma endregion

  // コンストラクタ
  explicit DynamicVector(const int size, const T initVal = {})
      : m_bufSize(size), m_initValue(initVal) {}

  // バッファ取得
  BufferType& Get() {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_storeBuffers.empty()) {
      m_buffers.emplace_front(m_bufSize, m_initValue);
      m_storeBuffers.emplace_back(m_buffers.front());
    }

    auto& buffer = m_storeBuffers.back().get();
    m_storeBuffers.pop_back();

    return buffer;
  }

  // バッファ解放
  void Release(BufferType& buf) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_storeBuffers.emplace_back(buf);
  }

  // バッファサイズ取得
  int GetBufferSize() const { return m_bufSize; }

  // バッファサイズ設定
  void SetBufferSize(const int size, const T initVal = {}) {
    m_bufSize = size;
    m_initValue = initVal;
    m_storeBuffers.clear();
    m_buffers.clear();
  }

  // サイズ取得
  int GetSize() const { return m_buffers.size(); }

  class Scoped {
    DynamicVector& m_dynamincBuf;

   public:
    BufferType& Buf;

#pragma region デフォルトメソッド定義

    /// <summary>
    /// デフォルトコンストラクタ
    /// </summary>
    Scoped() = delete;

    /// <summary>
    /// コピーコンストラクタ
    /// </summary>
    Scoped(const Scoped&) = delete;

    /// <summary>
    /// 代入演算子
    /// </summary>
    Scoped& operator=(const Scoped&) = delete;

    /// <summary>
    /// ムーブコンストラクタ
    /// </summary>
    Scoped(Scoped&&) = default;

    /// <summary>
    /// ムーブ代入演算子
    /// </summary>
    Scoped& operator=(Scoped&&) = default;

    /// <summary>
    /// デストラクタ
    /// </summary>
    ~Scoped() { m_dynamincBuf.Release(Buf); }

#pragma endregion

    /// <summary>
    /// コンストラクタ
    /// </summary>
    explicit Scoped(DynamicVector& dynamicBuf)
        : m_dynamincBuf(dynamicBuf), Buf(m_dynamincBuf.Get()) {}
  };
};

}  // namespace commonutility
