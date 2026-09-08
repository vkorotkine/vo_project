#pragma once

#include "core/Config.hpp"
#include "core/Frontend.hpp"
#include "types/Lie.hpp"
#include "types/Types.hpp"
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <optional>
#include <utility>
#include <vector>
namespace slam_core {

struct DownsampleResult {
  slam_types::ImageFeatures image_feats;
  std::vector<int>
      feat_idx; // which pixel each image feature corresponds to in the original
};

slam_types::ImageFeatures
downsample_to_grid(const slam_types::ImageFeatures &feats,
                   int subsample_grid_height, int subsample_grid_width,
                   int num_pts_per_cell);

// Camera vs X forward convention caused bugs, so taking it out for now
// inline Eigen::Matrix3d C_ZFtoXF() {
//   // Camera frame is z forward, x right, y down.
//   // Typical SLAM frame is
//   // x forward, y left, z up.
//   // So we convert at the end.
//   // x -> -y, y -> -z, z-> x

//   Eigen::Matrix3d C_C1toC; // where C is x-forward camera frame.
//   C_C1toC << 0, 0, 1, -1, 0, 0, 0, -1, 0;
//   return C_C1toC;
// }
inline Eigen::Vector3d backproject(const Eigen::Vector2d &uv, double depth,
                                   slam_types::CameraIntrinsics intrinsics) {
  // depth in meters. uv in pixels.
  // returns relative point position in camera frame.
  // Note on reference frames.
  // Camera frame is z forward, x right, y down.
  // Typical SLAM frame is
  // x forward, y left, z up.
  // So we convert at the end.
  // x -> -y, y -> -z, z-> x

  // Wait so we also need to undistort right.
  // Do we do this here or beforehand??

  // Camera frame convention
  double x = (uv[0] - intrinsics.cx) / intrinsics.fx * depth;
  double y = (uv[1] - intrinsics.cy) / intrinsics.fy * depth;
  Eigen::Vector3d p_LinC1 = Eigen::Vector3d{x, y, depth};
  return p_LinC1;
}

struct PnPResult {
  std::optional<lie::SE3> T;
  slam_utils::Bimap<size_t, slam_types::LandmarkId>
      matched_correspondences; // Feature Index (this Result is for a particular
                               // ImageFeature frame)
  std::vector<int>
      unmatched_features; // Indices of features that remain unmatched
  bool ok() {
    bool ret = true;
    if (!T)
      ret = false;
    return ret;
  }
};

class Estimator {

public:
  using SE3State = lie::State<lie::SE3>;
  explicit Estimator(slam_core::EstimatorOptions estimator_opts_,
                     slam_types::CameraIntrinsics intrinsics_)
      : opts(estimator_opts_), intrinsics(intrinsics_) {
    initialized = false;
  };
  std::optional<SE3State>
  process_features(const slam_types::ImageFeatures &image_feats);
  const std::vector<slam_types::Keyframe> &get_keyframes() const {
    return keyframes;
  };

  slam_types::ImageFeatures
  downsample_to_grid_(const slam_types::ImageFeatures &feats) {
    return downsample_to_grid(feats, opts.downsample.subsample_grid_height,
                              opts.downsample.subsample_grid_width,
                              opts.downsample.grid_num_points_per_cell);
  }

  const std::unordered_map<slam_types::LandmarkId, slam_types::Landmark> &
  get_landmarks() {
    return landmarks;
  }

private:
  slam_core::EstimatorOptions opts;
  std::vector<slam_types::Keyframe> keyframes;
  std::unordered_map<slam_types::KeyframeId, int> kf_id_to_index;
  int num_frames_since_last_kf = 0;

  slam_types::LandmarkId latest_landmark_id{0};
  std::unordered_map<slam_types::LandmarkId, slam_types::Landmark> landmarks;

  bool initialized;
  cv::BFMatcher matcher{cv::NORM_HAMMING};
  slam_types::CameraIntrinsics intrinsics;
  Eigen::Vector3d backproject_(const Eigen::Vector2d &uv, double depth) {
    return backproject(uv, depth, intrinsics);
  }
  void initialize_landmarks(slam_types::Keyframe &kf,
                            const slam_types::ImageFeatures &image_feats);
  std::unordered_map<int, int> match(cv::Mat query_descriptors,
                                     cv::Mat train_descriptors,
                                     bool mutual_consistency, float lowe_ratio);
  void add_keyframe(const slam_types::ImageFeatures &image_feats);
  PnPResult global_pose_PnP(const slam_types::ImageFeatures &image_feats,
                            const slam_types::Keyframe &kf);
};

} // namespace slam_core