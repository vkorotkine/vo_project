#pragma once

#include <array>
#include <core/Config.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <optional>
#include <stdexcept>
#include <types/Types.hpp>
#include <utility>

// Notes
// For invalid depth, the dataset sets the corresponding depth map entry to NaN.
// Depth map units are converted to meters.

namespace slam_io {

inline std::ostream &
operator<<(std::ostream &os,
           const std::optional<slam_types::Frame> &opt_frame) {
  if (opt_frame)
    os << *opt_frame;
  else
    os << "Empty frame" << std::endl;

  return os;
}

class Dataset {
public:
  using Ptr = std::unique_ptr<Dataset>;
  virtual ~Dataset() = default;
  virtual std::optional<slam_types::Frame> next() = 0;
  virtual slam_types::CameraIntrinsics intrinsics() const = 0;
};

struct Entry {
  double stamp;
  std::string rgb_path;
  std::string depth_path;
};

inline std::ostream &operator<<(std::ostream &os, const Entry &entry) {
  os << "Entry. Stamp: " << entry.stamp << " RGB path " << entry.rgb_path
     << " Depth path " << entry.depth_path;
  return os;
}

class TumDataset : public Dataset {
public:
  ~TumDataset() override = default;
  std::optional<slam_types::Frame> next() override;

  slam_types::CameraIntrinsics intrinsics() const override;

  explicit TumDataset(const slam_core::DataOptions &data_options);
  std::vector<std::pair<double, std::string>>
  readStampPathFile(const std::filesystem::path &path) {
    std::vector<std::pair<double, std::string>>
        img_entries; // just stamp + path
    std::ifstream file(path);
    std::string line;

    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#')
        continue;
      double timestamp;
      std::string fname;
      std::istringstream iss(line);
      if (iss >> timestamp >> fname) {
        auto pair = std::make_pair(timestamp, fname);
        img_entries.push_back(pair);
      }
    }
    return img_entries;
  }

  const std::vector<Entry> &get_entries() const { return entries; }
  static slam_types::Frame load_frame(double stamp,
                                      std::filesystem::path rgb_path,
                                      std::filesystem::path depth_path);

private:
  std::size_t counter;
  std::vector<Entry> entries;
  std::filesystem::path ds_path;
};

inline std::ostream &operator<<(std::ostream &os, const TumDataset &ds) {
  os << "Tum Dataset with " << ds.get_entries().size() << " entries"
     << std::endl;
  os << "First entry: " << ds.get_entries().front().stamp << " "
     << ds.get_entries().front().rgb_path << ds.get_entries().front().depth_path
     << std::endl;
  os << "Last entry: " << ds.get_entries().back().stamp << " "
     << ds.get_entries().back().rgb_path << ds.get_entries().back().depth_path
     << std::endl;
  return os;
}

std::unique_ptr<Dataset> dataset_factory(slam_core::DataOptions data_options);

} // namespace slam_io