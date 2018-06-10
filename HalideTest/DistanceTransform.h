#pragma once

template <typename T = double, typename P = tbb::auto_partitioner>
class DistanceTransform_ {
 public:
  using PartitionerType = P;

 private:
  //! バッファは浮動小数点型のみ
  static_assert(std::is_floating_point<T>::value,
                "template parameter T must be floatng point type");

  //
  // メンバ変数
  //

  //! 画像のサイズ
  cv::Size m_size = {};
  //! 計算結果(2乗距離)
  std::vector<T> m_sqDist = {};
  //! インデックス(X)
  std::vector<int> m_idxX = {};
  //! インデックス(Y)
  std::vector<int> m_idxY = {};
  //! 二乗距離バッファ
  std::vector<T> m_bufSqDist = {};
  //! 計算用バッファ
  std::vector<int> m_bufIdx = {};
  //! 計算用バッファ
  std::vector<T> m_bufZ = {};

  //! パーティショナー群
  struct Partitioners {
    PartitionerType Pa = {};
    PartitionerType X = {};
    PartitionerType Y = {};
  };
  //! パーティショナー群
  Partitioners m_partitioners = {};
  
    /// <summary>
  /// 1次元の距離変換
  /// </summary>
  /// <param name="size">[in] サイズ</param>
  /// <param name="step">[in] ステップ</param>
  /// <param name="f">[in] 入力データ</param>
  /// <param name="d">[out] バッファ</param>
  /// <param name="i">[out] バッファ</param>
  /// <param name="v">[out] バッファ</param>
  /// <param name="z">[out] バッファ</param>
  template <typename Tb>
  static void Transform1D(const int size, const int step, const Tb* const f, Tb* const d,
                          int* const i, int* const v, Tb* const z) {
    const Tb* ptrF1 = f + 1;
    int* ptrV0 = v;
    Tb* ptrZ0 = z;
    Tb* ptrZ1 = z + 1;

    *ptrV0 = 0;
    *ptrZ0 = std::numeric_limits<Tb>::lowest();
    *ptrZ1 = std::numeric_limits<Tb>::max();
    for (int q = 1; q < size; ++q, ++ptrF1) {
      const Tb A = (*ptrF1) + (q * q);
      const int& v1 = *ptrV0;
      Tb s = (A - (f[v1] + (v1 * v1))) / ((q - v1) << 1);
      while (s <= *ptrZ0) {
        --ptrV0;
        --ptrZ0;

        const int& v2 = *ptrV0;
        s = (A - (f[v2] + (v2 * v2))) / ((q - v2) << 1);
      }
      ++ptrV0;
      ++ptrZ0;

      *ptrV0 = q;
      *ptrZ0 = s;
      *(ptrZ0 + 1) = std::numeric_limits<Tb>::max();
    }

    Tb* ptrD0 = d;
    int* ptrI0 = i;
    ptrV0 = v;
    for (int q = 0; q < size; ++q, ptrD0 += step, ++ptrI0) {
      while (*ptrZ1 < q) {
        ++ptrV0;
        ++ptrZ1;
      }
      const int& refV = *ptrV0;
      *ptrD0 = commonutility::SquareNorm(q - refV) + f[refV];
      *ptrI0 = refV;
    }
  }

  /// <summary>
  /// 2次元の距離変換
  /// </summary>
  /// <param name="size">[int] 画像サイズ</param>
  /// <param name="data">[int,out] 変換前後のデータ</param>
  /// <param name="idxX">[out] 最短座標(X)</param>
  /// <param name="idxY">[out] 最短座標(Y)</param>
  /// <param name="work">[out] バッファ</param>
  /// <param name="workV">[out] バッファ</param>
  /// <param name="workZ">[out] バッファ</param>
  template <typename Tb>
  static void Transform2D(const cv::Size& size, std::vector<Tb>& data, std::vector<int>& idxX,
                          std::vector<int>& idxY, std::vector<Tb>& work, std::vector<int>& workV,
                          std::vector<Tb>& workZ, PartitionerType& paX, PartitionerType& paY) {
    // 横方向の距離変換
    tbb::tbb_for(0, size.height,
                 [&](const int y) {
                   const int idx = y * size.width;

                   // ※vs2010のラムダ式の不具合により、template引数も記述
                   DistanceTransform_<T, P>::Transform1D(size.width, size.height, &data[idx],
                                                         &work[y], &idxX[idx], &workV[idx],
                                                         &workZ[idx + y]);
                 },
                 paX);

    // 縦方向の距離変換
    tbb::tbb_for(0, size.width,
                 [&](const int x) {
                   const int idx = x * size.height;

                   // ※vs2010のラムダ式の不具合により、template引数も記述
                   DistanceTransform_<T, P>::Transform1D(size.height, size.width, &work[idx],
                                                         &data[x], &idxY[idx], &workV[idx],
                                                         &workZ[idx + x]);
                 },
                 paY);
  }

  /// <summary>
  /// 二値化&距離変換(二乗距離)
  /// </summary>
  /// <param name="src">[in] 入力</param>
  /// <param name="comp">[in] 二値化関数</param>
  /// <returns>変換結果</returns>
  /// <remarks>平方根をしてない結果を返すメソッド</remarks>
  template <typename Ts, typename F>
  const cv::Mat TransformSqDist(const cv::Mat& src, const F& comp) {
    Assert(src.channels() == 1 && src.size() == m_size, "Src-Data Error");

    // 2値データを浮動小数点型に変換
    cv::Mat buf(m_size, cv::Type<T>(), &m_sqDist[0]);
    tbb::tbb_for(src.size(),
                 [&](const tbb::blocked_range2d<int>& range) {
                   for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
                     int x = std::begin(range.cols());
                     const Ts* ptrSrc = src.ptr<Ts>(y, x);
                     T* ptrBuf = buf.ptr<T>(y, x);
                     for (; x < std::end(range.cols()); ++x, ++ptrSrc, ++ptrBuf)
                       *ptrBuf = comp(*ptrSrc) ? std::numeric_limits<T>::max() : T(0);
                   }
                 },
                 m_partitioners.Pa);

    // 変換
    DistanceTransform_::Transform2D(m_size, m_sqDist, m_idxX, m_idxY, m_bufSqDist, m_bufIdx, m_bufZ,
                                    m_partitioners.X, m_partitioners.Y);

    return buf;
  }

  /// <summary>
  /// 出力データ変換
  /// </summary>
  /// <param name="sqDist">[in] 二乗距離</param>
  /// <param name="dst">[out] 出力</param>
  /// <param name="conv">[in] 変換関数</param>
  /// <remarks>計算結果を出力型に変換する</remarks>
  template <typename Td, typename F>
  void ConvertTo(const cv::Mat& sqDist, cv::Mat& dst, const F& conv) {
    Assert(dst.channels() == 1 && dst.size() == m_size, "Dst-Data Error");

    // 出力データに変換
    tbb::tbb_for(dst.size(),
                 [&](const tbb::blocked_range2d<int>& range) {
                   for (int y = std::begin(range.rows()); y < std::end(range.rows()); ++y) {
                     int x = std::begin(range.cols());
                     const T* ptrSqDist = sqDist.ptr<T>(y, x);
                     Td* ptrDst = dst.ptr<Td>(y, x);
                     for (; x < std::end(range.cols()); ++x, ++ptrSqDist, ++ptrDst)
                       *ptrDst = conv(*ptrSqDist);
                   }
                 },
                 m_partitioners.Pa);
  }
  
 public:

  // 初期化
  void Init(const cv::Size& size) {
    m_size = size;

    const size_t bufSize = m_size.area();
    m_sqDist.resize(bufSize);
    m_idxX.resize(bufSize);
    m_idxY.resize(bufSize);
    m_bufSqDist.resize(bufSize);
    m_bufIdx.resize(bufSize);
    m_bufZ.resize(bufSize + std::max(m_size.width, m_size.height));
  }
  
  // 距離変換
  template <typename Ts, typename Td>
  void CalcCompCast(const cv::Mat& src, cv::Mat& dst,
                    const std::function<const bool(const Ts&)>& comp,
                    const std::function<const Td(const T&)>& cast) {
    this->ConvertTo<Td>(this->TransformSqDist<Ts>(src, comp), dst, cast);
  }
};
  
