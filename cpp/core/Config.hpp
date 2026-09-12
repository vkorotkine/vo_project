#pragma once

#include <ostream>
#include <string>
#include <types/Types.hpp>

namespace slam_core {
// TODO Have to expose downsample settings to constructor :)
struct EstimatorOptions {
public:
  bool test_mode = true;
  struct RansacPnP {
    int iterationsCount = 100;
    float reprojectionError = 3.0F;
    double confidence = 0.999;
  };

  struct Downsample {
    int subsample_grid_width = 32;
    int subsample_grid_height = 32;
    int grid_num_points_per_cell = 20;
  };

  struct BundleAdjustement {
    bool enabled = true;
    // in keyframes
    int window_size = 10;
    int run_every_num_frames = 5;
    int start_running_at_keyframe =
        10; // after how many keyframes we start running this
    double fake_baseline_rgbd = 0.3;
    double stereo_noise = 5;
    // How far away do keyframes have to be
    // that optimzie landmark, for landmark to get added to problem
    double parallax_threshold = 0.01;
    double pixel_noise_stdev = 1;
    double prior_noise_stdev = 0.001;
    int max_iter = 8;
  };

  // If we move a lot, new keyframe.
  // If a lot of frames passed since last frame, new keyframe.
  // If too few RANSAC inliers, new frame.
  struct KeyFrameInsertion {
    int frames_since_last_kf = 30;
    double covisibility_ratio =
        0.7; // once covisibility drops beneath this, new keyframe.
    int ransac_inlier_count = 50;
  };

  struct LandmarkCulling {

    // if we see a landmark, but dont see it at least
    // min_num_observations times in the span of
    // num_frames_before_drop frames,
    // we drop it.
    int num_frames_before_drop = 10;
    int min_num_observations = 2;
  };

  struct Matching {
    bool mutual_consistency = false;
    double lowe_ratio = 0.95;
    // bool mutual_consistency = true;
    // double lowe_ratio = 0.99;
  };

  EstimatorOptions(std::string estimator_type_,
                   int init_num_valid_depth_features_, std::string output_dir_)
      : estimator_type(estimator_type_),
        init_num_valid_depth_features(init_num_valid_depth_features_),
        output_dir(output_dir_){};

  std::string estimator_type;
  int init_num_valid_depth_features;
  std::string output_dir;
  RansacPnP ransac_pnp;
  Downsample downsample;
  KeyFrameInsertion keyframe_opts;
  Matching matching_opts;
  LandmarkCulling lndmrk_culling;
  BundleAdjustement bundle_adjustement;
};

struct DataOptions {
public:
  std::string path;
  std::string type;
  double association_tol;

  DataOptions(const std::string &path_, const std::string &type_,
              double association_tol_)
      : path(path_), type(type_), association_tol(association_tol_){};
};

struct FrontendOptions {
  // how many orb features to extract per image
  int num_raw_orb_features;
  // filter out points further than this
  double depth_threshold;
  // edge discontinuity rejection
  // construct square around feature, compute depth variance
  // reject for initialization if higher than threshold
  int feature_variance_square_dim;
  double max_feature_stddev;
  // subsampling

  // Initialization validity.
  FrontendOptions(int num_raw_orb_features_, double depth_threshold_,
                  int feature_variance_square_dim_, double max_feature_stddev_)
      : num_raw_orb_features(num_raw_orb_features_),
        depth_threshold(depth_threshold_),
        feature_variance_square_dim(feature_variance_square_dim_),
        max_feature_stddev(max_feature_stddev_){};
};

struct Config {
  Config(EstimatorOptions estimator_options_, DataOptions data_options_,
         FrontendOptions frontend_options_)
      : estimator_options(estimator_options_), data_options(data_options_),
        frontend_options(frontend_options_){};
  EstimatorOptions estimator_options;
  DataOptions data_options;
  FrontendOptions frontend_options;
};

inline std::ostream &operator<<(std::ostream &os,
                                const EstimatorOptions &estimator_options) {
  os << "  Estimator Options " << std::endl;
  os << "  estimator_type: " << estimator_options.estimator_type << std::endl;
  os << "  ransac confidence: " << estimator_options.ransac_pnp.confidence
     << std::endl;
  os << "  ransac iterations: " << estimator_options.ransac_pnp.iterationsCount
     << std::endl;
  os << "  ransac reproj error: "
     << estimator_options.ransac_pnp.reprojectionError << std::endl;
  os << " downsample num pts per cell"
     << estimator_options.downsample.grid_num_points_per_cell << std::endl;
  os << " downsample cell height"
     << estimator_options.downsample.subsample_grid_height << std::endl;
  os << " downsample cell width"
     << estimator_options.downsample.subsample_grid_width << std::endl;
  return os;
}

inline std::ostream &operator<<(std::ostream &os,
                                const DataOptions &data_options) {
  os << "  Data Options " << std::endl;
  os << "  path: " << data_options.path << std::endl;
  os << "  type: " << data_options.type << std::endl;
  os << "  association_tol: " << data_options.association_tol << std::endl;
  double association_tol;
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const FrontendOptions &opts) {
  os << "  Frontend Options " << std::endl;
  os << "  num_raw_orb_features: " << opts.num_raw_orb_features << std::endl;
  os << "  depth_threshold: " << opts.depth_threshold << std::endl;
  os << "  feature_variance_square_dim: " << opts.feature_variance_square_dim
     << std::endl;
  os << "  max_feature_stddev: " << opts.max_feature_stddev << std::endl;
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const Config &config) {

  os << "Config Options" << std::endl;
  os << config.estimator_options;
  os << config.data_options;
  os << config.frontend_options;
  return os;
}

} // namespace slam_core
