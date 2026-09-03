#include <core/Config.hpp>
#include <filesystem>
#include <io/Dataset.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <stdexcept>
#include <types/Types.hpp>
#include <utility>

namespace slam_io {

std::unique_ptr<Dataset> dataset_factory(slam_core::DataOptions data_options) {
  if (data_options.type == "tum") {
    return std::make_unique<TumDataset>(data_options);
  }
  throw std::runtime_error("Dataset is not supported!");
}

slam_types::Frame TumDataset::load_frame(double stamp,
                                         std::filesystem::path rgb_path,
                                         std::filesystem::path depth_path) {
  cv::Mat rgb = cv::imread(rgb_path, cv::IMREAD_COLOR);
  cv::Mat depth_raw = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
  cv::Mat depth;

  depth_raw.convertTo(depth, CV_32F, 1.0 / 5000.0);
  cv::Mat mask = (depth_raw == 0);
  depth.setTo(std::numeric_limits<float>::quiet_NaN(), mask);

  if (rgb.empty() || depth.empty())
    throw std::runtime_error("Empty images!");
  return slam_types::Frame(stamp, rgb, depth);
}

std::optional<slam_types::Frame> TumDataset::next() {
  if (counter >= entries.size())
    return std::nullopt;

  std::filesystem::path rgb_path = ds_path / entries.at(counter).rgb_path;
  std::filesystem::path depth_path = ds_path / entries.at(counter).depth_path;
  slam_types::Frame ret =
      load_frame(entries.at(counter).stamp, rgb_path, depth_path);
  counter++;
  return ret;
}

slam_types::CameraIntrinsics TumDataset::intrinsics() const {
  slam_types::CameraIntrinsics intr = slam_types::CameraIntrinsics(
      100, 100, 50, 50, 200, 200, std::array<double, 5>{0, 0, 0, 0, 0});
  return intr;
};

TumDataset::TumDataset(const slam_core::DataOptions &data_options) {
  counter = 0;

  ds_path = std::filesystem::path(data_options.path);
  std::filesystem::path rgb_path = ds_path / "rgb.txt";
  std::filesystem::path depth_path = ds_path / "depth.txt";
  std::filesystem::path gt_path = ds_path / "groundtruth.txt";

  if (!std::filesystem::exists(rgb_path)) {
    std::cout << "RGB Path: " << rgb_path << std::endl;
    throw std::runtime_error("No RGB file! %s");
  }
  if (!std::filesystem::exists(depth_path))
    throw std::runtime_error("No Depth file!");

  std::vector<std::pair<double, std::string>> rgb_entries;
  std::vector<std::pair<double, std::string>> depth_entries;
  std::vector<SE3State> states;

  rgb_entries = readStampPathFile(rgb_path);
  depth_entries = readStampPathFile(depth_path);
  states = readPoseFile(gt_path);

  for (const auto &rgb_entry : rgb_entries) {
    double rgb_timestamp = rgb_entry.first;
    auto closest =
        std::min_element(depth_entries.begin(), depth_entries.end(),
                         [rgb_timestamp](const auto &a, const auto &b) {
                           return (std::abs(a.first - rgb_timestamp) <
                                   std::abs(b.first - rgb_timestamp));
                         }

        );

    double dt = std::abs(closest->first - rgb_timestamp);
    if (dt > data_options.association_tol)
      continue;

    auto closest_gt =
        std::min_element(states.begin(), states.end(),
                         [rgb_timestamp](const SE3State &a, const SE3State &b) {
                           return (std::abs(a.stamp - rgb_timestamp) <
                                   std::abs(b.stamp - rgb_timestamp));
                         });
    double dt_gt = std::abs(closest_gt->stamp - rgb_timestamp);
    if (dt_gt > data_options.association_tol)
      return;

    Entry entry = Entry{rgb_entry.first, rgb_entry.second, closest->second,
                        closest_gt->x};
    // std::cout << entry << std::endl;
    entries.push_back(entry);
  }
  if (entries.size() == 0)
    throw std::runtime_error("Entries are empty!");
  std::cout << *this;
}

} // namespace slam_io