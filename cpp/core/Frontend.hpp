#pragma once

#include "io/Dataset.hpp"
#include <Eigen/Core>
#include <array>
#include <core/Config.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <optional>
#include <stdexcept>
#include <types/Types.hpp>
#include <utility>

namespace slam_core {

struct GridIndexer {
  GridIndexer(int ncols_) : ncols(ncols_){};

  int ncols;
  int coord_to_idx(int x, int y) { return y * ncols + x; }
  auto idx_to_coord(int idx) {
    return std::make_pair(idx % ncols, idx / ncols);
  }
};
// Extract descriptors.
// If depth validity criterion is not met, the corresponding depth value is set
// to NaN.
slam_types::ImageFeatures
extract_descriptors(const slam_types::Frame &frame, int num_descriptors,
                    int feature_variance_square_dim, double max_feature_stddev,
                    double max_depth, cv::Ptr<cv::ORB> orb);

class Frontend {
public:
  explicit Frontend(slam_core::FrontendOptions opts_) : opts(opts_) {
    orb = cv::ORB::create(opts.num_raw_orb_features);
  };

  slam_types::ImageFeatures
  extract_descriptors_(const slam_types::Frame &frame) {
    return extract_descriptors(
        frame, opts.num_raw_orb_features, opts.feature_variance_square_dim,
        opts.max_feature_stddev, opts.depth_threshold, orb);
  }

private:
  cv::Ptr<cv::ORB> orb;
  slam_core::FrontendOptions opts;
};

} // namespace slam_core