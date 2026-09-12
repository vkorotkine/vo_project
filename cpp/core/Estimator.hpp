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

slam_types::ImageFeatures
downsample_to_grid(const slam_types::ImageFeatures &feats,
                   int subsample_grid_height, int subsample_grid_width,
                   int num_pts_per_cell);

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
  void cullLandmarks();
  void newLandmarksFromUnmatched(const slam_types::ImageFeatures &unmatched,
                                 const lie::SE3 &T_CtoW,
                                 const slam_types::KeyframeId &kf_id);
  void bundleAdjustment();

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
    return intrinsics.backproject(uv, depth);
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