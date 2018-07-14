#pragma once

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <unordered_map>

#include <boost/config.hpp>
#include <boost/serialization/archive_input_unordered_map.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/unordered_collections_load_imp.hpp>
#include <boost/serialization/unordered_collections_save_imp.hpp>
#include <boost/serialization/utility.hpp>

// 警告抑制解除
MSVC_WARNING_POP

namespace boost { namespace serialization {

template <typename Archive, typename Key, typename T, typename HashFcn, typename EqualKey,
          typename Allocator>
inline void save(Archive &ar, const std::unordered_map<Key, T, HashFcn, EqualKey, Allocator> &t,
                 const unsigned) {
  boost::serialization::stl::save_unordered_collection<
      Archive, std::unordered_map<Key, T, HashFcn, EqualKey, Allocator> >(ar, t);
}

template <typename Archive, typename Key, typename T, typename HashFcn, typename EqualKey,
          typename Allocator>
inline void load(Archive &ar, std::unordered_map<Key, T, HashFcn, EqualKey, Allocator> &t,
                 const unsigned) {
  boost::serialization::stl::load_unordered_collection<
      Archive, std::unordered_map<Key, T, HashFcn, EqualKey, Allocator>,
      boost::serialization::stl::archive_input_unordered_map<
          Archive, std::unordered_map<Key, T, HashFcn, EqualKey, Allocator> > >(ar, t);
}

template <typename Archive, typename Key, typename T, typename HashFcn, typename EqualKey,
          typename Allocator>
inline void serialize(Archive &ar, std::unordered_map<Key, T, HashFcn, EqualKey, Allocator> &t,
                      const unsigned file_version) {
  boost::serialization::split_free(ar, t, file_version);
}

template <typename Archive, typename Key, typename T, typename HashFcn, typename EqualKey,
          typename Allocator>
inline void save(Archive &ar,
                 const std::unordered_multimap<Key, T, HashFcn, EqualKey, Allocator> &t,
                 const unsigned) {
  boost::serialization::stl::save_unordered_collection<
      Archive, std::unordered_multimap<Key, T, HashFcn, EqualKey, Allocator> >(ar, t);
}

template <typename Archive, typename Key, typename T, typename HashFcn, typename EqualKey,
          typename Allocator>
inline void load(Archive &ar, std::unordered_multimap<Key, T, HashFcn, EqualKey, Allocator> &t,
                 const unsigned) {
  boost::serialization::stl::load_unordered_collection<
      Archive, std::unordered_multimap<Key, T, HashFcn, EqualKey, Allocator>,
      boost::serialization::stl::archive_input_unordered_multimap<
          Archive, std::unordered_multimap<Key, T, HashFcn, EqualKey, Allocator> > >(ar, t);
}

template <typename Archive, typename Key, typename T, typename HashFcn, typename EqualKey,
          typename Allocator>
inline void serialize(Archive &ar, std::unordered_multimap<Key, T, HashFcn, EqualKey, Allocator> &t,
                      const unsigned file_version) {
  boost::serialization::split_free(ar, t, file_version);
}

}}  // namespace boost::serialization

// 全ての警告を抑制
MSVC_ALL_WARNING_PUSH

#include <cstddef>

#include <sstream>
#include <string>
#include <type_traits>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/nvp.hpp>

// 警告抑制解除
MSVC_WARNING_POP

/// <summary>
/// ユーティリティ
/// </summary>
namespace commonutility {

template <typename Archive, typename T,
          std::enable_if_t<!std::is_same<Archive, boost::archive::xml_oarchive>{} &&
                               !std::is_same<Archive, boost::archive::xml_iarchive>{},
                           std::nullptr_t> = nullptr>
void Serialize(Archive &ar, const char *, T &itm) {
  ar &itm;
}
template <typename Archive, typename T,
          std::enable_if_t<std::is_same<Archive, boost::archive::xml_oarchive>{}, std::nullptr_t> =
              nullptr>
void Serialize(Archive &ar, const char *tag, T &itm) {
  // 一旦文字列化
  std::ostringstream ss;
  boost::archive::binary_oarchive oa(ss);
  oa << itm;
  // 文字列としてシリアライズ
  ar << boost::serialization::make_nvp(tag, ss.str());
}
template <typename Archive, typename T,
          std::enable_if_t<std::is_same<Archive, boost::archive::xml_iarchive>{}, std::nullptr_t> =
              nullptr>
void Serialize(Archive &ar, const char *tag, T &itm) {
  // 文字列としてデシリアライズ
  std::string is;
  ar >> boost::serialization::make_nvp(tag, is);
  // 文字列から復元
  std::istringstream ss(is);
  boost::archive::binary_iarchive ia(ss);
  ia >> itm;
}

}  // namespace commonutility
