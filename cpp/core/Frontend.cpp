#include <algorithm>
#include <core/Config.hpp>
#include <core/Frontend.hpp>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <stdexcept>
#include <utility>

namespace slam_core {
slam_types::ImageFeatures
extract_descriptors(const slam_types::Frame &frame, int num_descriptors,
                    int feature_variance_square_dim, double max_feature_stddev,
                    double max_depth, cv::Ptr<cv::ORB> orb) {
  // Extract descriptors.
  // If depth validity criterion is not met, the corresponding depth value is
  // set to NaN

  // Extract Descriptors
  std::vector<cv::KeyPoint> keypoints;
  std::vector<double> depths;
  cv::Mat descriptors;
  orb->detectAndCompute(frame.rgb, cv::noArray(), keypoints, descriptors);
  std::cout << "Computed " << descriptors.rows << " descriptors" << std::endl;

  std::vector<Eigen::Vector2d> uv;

  for (const auto &kp : keypoints) {
    uv.emplace_back(kp.pt.x, kp.pt.y);

    int loc_y = std::lround(kp.pt.y);
    int loc_x = std::lround(kp.pt.x);

    bool discard_depth = false;
    // keypoint rounds to integer outside image. discard depth at that location.
    if (loc_y >= frame.rgb.rows || loc_x >= frame.rgb.cols || loc_x < 0 ||
        loc_y < 0)
      discard_depth = true;
    // keypoint centered patch
    int width = feature_variance_square_dim;
    cv::Rect patch(loc_x - width / 2, loc_y - width / 2, width, width);
    // intersection - clip to image size
    patch = patch & cv::Rect(0, 0, frame.depth.cols, frame.depth.rows);
    if (patch.width <= 0 || patch.height <= 0)
      discard_depth = true;
    else {
      cv::Scalar mean, stdDev;
      cv::meanStdDev(frame.depth(patch), mean, stdDev);
      if (std::isnan(stdDev[0]) || stdDev[0] > max_feature_stddev)
        discard_depth = true;
    }
    if (discard_depth)
      depths.push_back(std::numeric_limits<double>::quiet_NaN());
    else {
      double depth = double(frame.depth.at<float>(loc_y, loc_x));
      depths.push_back(depth);
    }
  }
  std::vector<int> ids(depths.size());
  std::iota(ids.begin(), ids.end(), 0);
  int num_remaining =
      std::count_if(depths.begin(), depths.end(),
                    [](const double &d) { return !std::isnan(d); });
  std::cout << "Num valid after depth filters: " << num_remaining << std::endl;

  if (keypoints.size() != uv.size()) {
    throw std::runtime_error("Keypoint and uv sizes dont match!");
  }

  if (ids.size() != uv.size()) {
    throw std::runtime_error("ID and uv sizes dont match!");
  }

  if (descriptors.rows != uv.size()) {
    throw std::runtime_error("Descriptor and uv sizes dont match!");
  }

  return slam_types::ImageFeatures{frame.timestamp, ids,         uv,
                                   depths,          descriptors, frame.rgb.cols,
                                   frame.rgb.rows};
}

// Need following functions
// Downsample to grid

// Remove features that have issues
// valid depth, depth range (set a threshold??), spatial distribution (grid),
// edges Check if there is sufficient info in there Initialize
// }

// cv::Mat Frontend::extract_descriptors(const slam_core::Frame &frame) {
//   std::vector<cv::KeyPoint> keypoints;
//   cv::Mat descriptors;
//   orb->detectAndCompute(frame.rgb, cv::noArray(), keypoints, descriptors);
// }

// Back project features wiht valid depth
// std::vector<cv::KeyPoint> temp_keypoints;
// std::vector<Eigen::Vector2d> temp_uv;

} // namespace slam_core