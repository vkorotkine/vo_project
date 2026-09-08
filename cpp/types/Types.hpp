#pragma once
#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "Utils.hpp"
#include "types/Lie.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

// TODO: Need T_BtoA notation
// T_ab inconsistent
namespace slam_types {
struct Frame {
  double timestamp;
  cv::Mat rgb;
  cv::Mat depth; // depth map in meters
  Frame(double timestamp_, const cv::Mat &rgb_,
        const cv::Mat &depth_ = cv::Mat(), const lie::SE3 &T_ab_ = lie::SE3())
      : timestamp(timestamp_), rgb(rgb_), depth(depth_), T_ab(T_ab_){};

  lie::SE3 T_ab; // camera to world.
};

inline std::ostream &operator<<(std::ostream &os,
                                const slam_types::Frame &frame) {
  double minval;
  double maxval;
  if (!frame.depth.empty()) {
    cv::minMaxLoc(frame.depth, &minval, &maxval);
    os << "Frame. Stamp: " << std::setprecision(6) << frame.timestamp << ", "
       << frame.rgb.rows << "rows by " << frame.rgb.cols << "columns. "
       << " Min depth (m): " << minval << " Max depth:" << maxval << std::endl;
  }
  return os;
}

// using LandmarkId = std::uint64_t;
// using KeyframeId = std::uint64_t;

enum class LandmarkId : std::uint64_t {};
enum class KeyframeId : std::uint64_t {};

template <typename T> inline T increment_id(T id) {
  T incremented = T{static_cast<std::uint64_t>(id) + 1};
  // std::cout << "Increment from " << static_cast<std::uint64_t>(id) << " to "
  //           << static_cast<std::uint64_t>(incremented) << std::endl;

  return incremented;
}

inline std::string to_string(LandmarkId id) {
  return "L" + std::to_string(static_cast<std::uint64_t>(id));
}

inline std::string to_string(KeyframeId id) {
  return "K" + std::to_string(static_cast<std::uint64_t>(id));
}

struct CameraIntrinsics {
  double fx, fy, cx, cy;
  int width, height;
  std::array<double, 5> distortion{0, 0, 0, 0, 0};

  Eigen::Matrix3d K() const {
    Eigen::Matrix3d m = Eigen::Matrix3d::Identity();
    m(0, 0) = fx;
    m(1, 1) = fy;
    m(0, 2) = cx;
    m(1, 2) = cy;
    return m;
  }

  CameraIntrinsics(double fx_, double fy_, double cx_, double cy_, int width_,
                   int height_, std::array<double, 5> distortion_)
      : fx(fx_), fy(fy_), cx(cx_), cy(cy_), width(width_), height(height_),
        distortion(distortion_){};
};

struct ImageFeature {
  double stamp;
  int id;
  Eigen::Vector2d uv;
  double depth;
  cv::Mat descriptor;
};

// right. this needs a feature ID. This can be an integer but it has to be kept
// track of throughout.

struct ImageFeatures {
  // uv coordinates: x, y. Horizontal then vertical.
  double stamp;
  std::vector<int> ids; // when we subsample, need to keep track of what id they
                        // were originally
  std::vector<Eigen::Vector2d> uv;
  std::vector<double> depths;
  cv::Mat descriptors;
  int img_width;
  int img_height;
  ImageFeatures(double stamp_, const std::vector<int> &ids_,
                const std::vector<Eigen::Vector2d> &uv_,
                const std::vector<double> &depths_, const cv::Mat &descriptors_,
                int img_width_, int img_height_)
      : stamp(stamp_), ids(ids_), uv(uv_), depths(depths_),
        descriptors(descriptors_), img_width(img_width_),
        img_height(img_height_) {
    if (uv.size() != depths.size() || depths.size() != descriptors.rows ||
        ids.size() != depths.size()) {
      std::cout << "IDs size" << ids.size() << "UV size: " << uv.size()
                << " Depth size: " << depths.size() << " Descriptor rows "
                << descriptors.rows << std::endl;
      throw std::runtime_error("Image Features constructed with uv size != "
                               "depths size!");
    }
  }
  size_t num_valid_depths() const {
    return std::count_if(depths.begin(), depths.end(),
                         [](double a) { return !std::isnan(a); });
  }
  ImageFeature get_single_feature(size_t idx) {
    ImageFeature feat =
        ImageFeature{stamp, ids.at(idx), uv.at(idx), depths.at(idx),
                     descriptors.row(idx).clone()};
    return feat;
  }
  ImageFeatures get_subset(std::vector<int> subset_indices) const {
    std::vector<Eigen::Vector2d> uv_;
    std::vector<double> depths_;
    std::vector<int> ids_;
    cv::Mat descriptors_;
    for (const auto &i : subset_indices) {
      ids_.push_back(ids.at(i));
      uv_.push_back(uv.at(i));
      depths_.push_back(depths.at(i));
      descriptors_.push_back(descriptors.row(i).clone());
    }
    return ImageFeatures(stamp, ids_, uv_, depths_, descriptors_, img_width,
                         img_height);
  }
};

inline std::ostream &operator<<(std::ostream &os, const ImageFeatures &feats) {
  os << "ImageFeatures. Stamp: " << feats.stamp
     << " Size uv vec: " << feats.uv.size()
     << " Descripts with size: " << feats.descriptors.rows << " by "
     << feats.descriptors.cols;
  return os;
}

// struct Observation {
//   LandmarkId landmark_id;
//   std::size_t feature_idx;
// };

struct Landmark {
  // Position: p_LinA, where A is world frame
  LandmarkId id;
  Eigen::Vector3d position;
  std::vector<KeyframeId> observed_in;
  int num_frames_since_last_obs = 0;
  Landmark(LandmarkId id_, Eigen::Vector3d position_,
           std::vector<KeyframeId> observed_in_)
      : id(id_), position(position_), observed_in(observed_in_){};
};

struct Keyframe {
  Keyframe(KeyframeId id_, lie::SE3 pose_, const ImageFeatures &features_,
           size_t num_landmarks)
      : id(id_), pose(pose_), features(features_) {
    Feat2Landmark = slam_utils::Bimap<size_t, LandmarkId>(num_landmarks);
  }
  Keyframe(KeyframeId id_, lie::SE3 pose_, const ImageFeatures &features_,
           const slam_utils::Bimap<size_t, LandmarkId> &feature_to_landmark_map)
      : id(id_), pose(pose_), features(features_) {
    Feat2Landmark = feature_to_landmark_map; // case where we already know a
                                             // feature to landmark map
  }

  void add_landmark_feature_correspondence(LandmarkId l_idx, size_t feat_idx) {
    Feat2Landmark.insert(feat_idx, l_idx);
  }
  void delete_landmark(LandmarkId l_idx) { Feat2Landmark.delete_value(l_idx); }

  LandmarkId feature_to_landmark(const size_t &feat_idx) const {
    return Feat2Landmark.forward(feat_idx);
  }
  size_t landmark_to_feature(const LandmarkId &l) const {
    return Feat2Landmark.reverse(l);
  }
  // bool landmark_observed(LandmarkId l){
  // return map.count()
  // }
  bool feature_is_landmark(size_t feat_idx) const {
    return Feat2Landmark.key_present(feat_idx);
  }
  double timestamp() const { return features.stamp; }

  KeyframeId id;

  lie::SE3 pose; // T_AtoK

  // raw
  ImageFeatures features;

  // This should probably be private. Being lazy with the printing.
  slam_utils::Bimap<size_t, LandmarkId> Feat2Landmark;
};

inline std::ostream &operator<<(std::ostream &os, const Keyframe &kf) {
  os << "Keyframe with Stamp " << kf.timestamp() << " Keyframe ID "
     << std::size_t(kf.id) << std::endl
     << kf.pose << std::endl
     << kf.features;
  os << kf.Feat2Landmark.size();
  //  << " observations. Random 10, (FT_ID, LNDMRK_ID) ";

  // int idx = 0;
  // for (const auto &[key, value] : kf.featIdx2Landmark)
  //   os << "(" << key << "," << std::size_t(value) << "),";

  return os;
}
} // namespace slam_types