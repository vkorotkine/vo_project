#pragma once

#include <map>
#include <stdexcept>
#include <string>

namespace slam_utils {
template <typename Map, typename K, typename V>
void insert_unique(Map &map, const K &key, const V &value) {
  using std::to_string;
  auto [it, ok] = map.emplace(key, value);
  if (!ok) {
    throw std::runtime_error("Duplicate key: " + to_string(key));
  }
}

template <typename FwdMap, typename RevMap, typename K, typename V>
void insert_unique_bimap(FwdMap &fwd, RevMap &rev, const K &key,
                         const V &value) {
  using std::to_string;
  auto [it, ok] = fwd.emplace(key, value);
  if (!ok) {
    throw std::runtime_error("Forward. Duplicate key: " + to_string(key) +
                             ", Value: " + to_string(value));
  }
  auto [it1, ok1] = rev.emplace(value, key);
  if (!ok1) {
    throw std::runtime_error("Backward. Duplicate value: " + to_string(value) +
                             ", Key: " + to_string(key));
  }
}

template <typename K, typename V> class Bimap {

public:
  Bimap() = default;
  Bimap(size_t num_values) { allocate(num_values); }
  void allocate(size_t num_values) {
    fwd_map.reserve(num_values);
    rev_map.reserve(num_values);
  }
  void insert(const K &key, const V &val) {
    insert_unique_bimap(fwd_map, rev_map, key, val);
  }
  V forward(const K &key) const { return fwd_map.at(key); }
  K reverse(const V &val) const { return rev_map.at(val); }
  bool key_present(const K &key) const { return fwd_map.count(key) > 0; }
  bool value_present(const V &val) const { return rev_map.count(val) > 0; }
  size_t size() const { return fwd_map.size(); }

  const std::unordered_map<K, V> &forward_map() const { return fwd_map; }
  const std::unordered_map<V, K> &reverse_map() const { return rev_map; }

private:
  std::unordered_map<K, V> fwd_map;
  std::unordered_map<V, K> rev_map;
};

} // namespace slam_utils