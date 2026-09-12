#include "core/Estimator.hpp"
#include "core/Frontend.hpp"
#include "io/Dataset.hpp"
#include "types/Lie.hpp"
#include "types/Types.hpp"
#include <Eigen/Core>
#include <filesystem>
#include <gtest/gtest.h>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <utility>

TEST(Estimator, GridIndexer) {
  using namespace slam_core;
  GridIndexer gi = GridIndexer(5);
  int x = 3;
  int y = 4;

  int idx = gi.coord_to_idx(x, y);
  std::pair<int, int> coord = gi.idx_to_coord(idx);
  EXPECT_EQ(x, coord.first);
  EXPECT_EQ(y, coord.second);
}

TEST(Estimator, TumImageTest) {
  using namespace slam_core;
  using namespace slam_io;
  using namespace slam_types;
  FrontendOptions frontend_options(1000, 5, 4, 0.2);

  Frontend frontend = Frontend(frontend_options);
  cv::Mat rgb =
      cv::imread(std::filesystem::path(TEST_DATA_DIR) / "tum_img.png");
  cv::Mat depths = cv::Mat::zeros(rgb.rows, rgb.cols, CV_64F);
  const Frame frame(0.5, rgb, depths);
  ImageFeatures feats = frontend.extract_descriptors_(frame);
  ImageFeatures feats_downsampled = downsample_to_grid(feats, 200, 30, 10);
}

TEST(Estimator, DownsampleToyTest) {
  using namespace slam_core;
  using namespace slam_types;

  int img_width = 300;
  int img_height = 100;
  int subsample_grid_width = 100;
  int subsample_grid_height = 30;
  int pts_per_cell = 2;

  std::vector<Eigen::Vector2d> uv;
  uv.emplace_back(51, 52);
  uv.emplace_back(53, 54);
  uv.emplace_back(55, 56);
  uv.emplace_back(120, 50);
  uv.emplace_back(121, 55);
  uv.emplace_back(122, 56);
  // expected result: 4 points returned. The first two of each cell.

  std::vector<double> depths{5., 5., 5., 5, 5, 5};
  cv::Mat descriptors(6, 8, CV_8UC1, cv::Scalar(0));

  std::vector<int> ids(depths.size());
  std::iota(ids.begin(), ids.end(), 0);

  ImageFeatures feats =
      ImageFeatures{1.0, ids, uv, depths, descriptors, img_width, img_height};

  ImageFeatures feats_downsampled = slam_core::downsample_to_grid(
      feats, subsample_grid_height, subsample_grid_width, pts_per_cell);

  EXPECT_EQ(feats_downsampled.uv.size(), 4);

  // Since its ordered we can also check exactly which ones are kept
  EXPECT_EQ(feats_downsampled.uv.at(0), Eigen::Vector2d(51, 52));
  EXPECT_EQ(feats_downsampled.uv.at(1), Eigen::Vector2d(53, 54));
  EXPECT_EQ(feats_downsampled.uv.at(2), Eigen::Vector2d(120, 50));
  EXPECT_EQ(feats_downsampled.uv.at(3), Eigen::Vector2d(121, 55));
}

TEST(Estimator, BackProject) {
  using namespace slam_core;
  using namespace slam_types;
  using SE3State = lie::State<lie::SE3>;

  CameraIntrinsics intrinsics = CameraIntrinsics(
      525, 525, 319.5, 239.5, 640, 480, std::array{0., 0., 0., 0., 0.});
  Config cfg = Config(
      EstimatorOptions("baseline", 130, "/home/vio_ws/src/vio_project/output"),
      DataOptions("/home/datasets/tum/rgbd_dataset_freiburg1_xyz", "tum", 0.2),
      FrontendOptions(1000, 5, 4, 0.2));
  Estimator estimator = Estimator(cfg.estimator_options, intrinsics);
  Eigen::Vector3d p_LinC{1, 2, 3};
  std::cout << p_LinC.transpose() << std::endl;
}
// As an integration test, we will later run this thing over
// whole dataset. Would need to load ground truth into DS too.
TEST(Estimator, PnPTUM) {
  using namespace slam_core;
  using namespace slam_types;
  using SE3State = lie::State<lie::SE3>;
  // Load all of our stuff
  std::string gt_line1 =
      "1305031102.1758 1.3405 0.6266 1.6575 0.6574 0.6126 -0.2949 -0.3248";
  std::string gt_line2 =
      "1305031127.9155 1.2993 0.5812 1.4485 0.6684 0.6485 -0.2797 -0.2335";

  // Now convert to Poses
  SE3State T_C1toW = SE3State::from_line(gt_line1);
  SE3State T_C2toW = SE3State::from_line(gt_line2);

  // Now the image frames
  std::filesystem::path img_test_dir =
      std::filesystem::path(TEST_DATA_DIR) / "pnp_test";
  slam_types::Frame frame1 = slam_io::TumDataset::load_frame(
      T_C1toW.stamp, img_test_dir / "1305031102.175304.png",
      img_test_dir / "depth_1305031102.160407.png");
  slam_types::Frame frame2 = slam_io::TumDataset::load_frame(
      T_C2toW.stamp, img_test_dir / "1305031127.911476.png",
      img_test_dir / "depth_1305031127.922560.png");

  // Set up Estimator
  // Test with same image to make sure we have identity.
  Config cfg = Config(
      EstimatorOptions("baseline", 130, "/home/vio_ws/src/vio_project/output"),
      DataOptions("/home/datasets/tum/rgbd_dataset_freiburg1_xyz", "tum", 0.2),
      FrontendOptions(1000, 5, 4, 0.2));
  std::cout << cfg;
  CameraIntrinsics intrinsics = CameraIntrinsics(
      525, 525, 319.5, 239.5, 640, 480, std::array{0., 0., 0., 0., 0.});
  Estimator estimator = Estimator(cfg.estimator_options, intrinsics);
  Frontend frontend = Frontend(cfg.frontend_options);

  // Test 1: Make sure that with same image being processed, the result is
  // identity transformation.
  ImageFeatures image_feats1 = frontend.extract_descriptors_(frame1);
  estimator.process_features(image_feats1);
  //   std::cout << image_feats2 << std::endl;
  auto res = estimator.process_features(image_feats1);
  if (!res)
    std::runtime_error("Processing features failed");

  SE3State T2 = *res;
  slam_types::Keyframe kf = estimator.get_keyframes().at(0);
  SE3State T1 = SE3State(kf.timestamp(), kf.pose);
  EXPECT_TRUE(T1.x.toMatrix().isApprox(T2.x.toMatrix(), 1e-8));
  EXPECT_TRUE(std::abs(T1.stamp - T2.stamp) < 1e-15);

  std::cout << "Test 2 PnP" << std::endl;
  // Test 2: Two images, ground truth, make sure pose transformation is similar.
  // would have been better to split this and not reusude variables but meh
  estimator = Estimator(cfg.estimator_options, intrinsics);
  estimator.process_features(image_feats1);
  ImageFeatures image_feats2 = frontend.extract_descriptors_(frame2);
  res = estimator.process_features(image_feats2);

  T2 = *res;
  kf = estimator.get_keyframes().at(0);
  T1 = SE3State(kf.timestamp(), kf.pose);

  // Test the PnP
  lie::SE3 dT_est = T1.x.Inverse() * T2.x;
  lie::SE3 dT_gt = T_C1toW.x.Inverse() * T_C2toW.x;

  Eigen::Vector3d dxi_phi =
      lie::Minus(dT_est.C, dT_gt.C, lie::Direction::Right);
  Eigen::Vector3d dr = dT_est.p - dT_gt.p;

  EXPECT_LE(dxi_phi.norm(), 0.03);
  EXPECT_LE(dr.norm(), 0.03);
}