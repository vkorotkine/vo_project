#include "core/Config.hpp"
#include "core/Estimator.hpp"
#include "core/Frontend.hpp"
#include "io/Dataset.hpp"
#include "types/Lie.hpp"
#include "types/Types.hpp"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <utility>

// TODO: Hardocded dataset path is not halal.
// Need to pass as option or something.
using namespace slam_core;
using namespace slam_io;
using namespace slam_types;

TEST(Integration, TUM) {
  using SE3State = lie::State<lie::SE3>;
  Config cfg = Config(
      EstimatorOptions("baseline", 130, "/home/vio_ws/src/vio_project/output"),
      DataOptions("/home/datasets/tum/rgbd_dataset_freiburg1_xyz", "tum", 0.2),
      FrontendOptions(1000, 3.5, 5, 0.1));
  std::cout << cfg;
  Dataset::Ptr ds = dataset_factory(cfg.data_options);
  CameraIntrinsics intrinsics = CameraIntrinsics(
      525, 525, 319.5, 239.5, 640, 480, std::array{0., 0., 0., 0., 0.});
  Estimator estimator = Estimator(cfg.estimator_options, intrinsics);
  Frontend frontend = Frontend(cfg.frontend_options);

  int num_frames = 30;
  int frame_idx = 0;

  SE3State prev_state{};
  lie::SE3 T_gt0;
  while (auto frame = ds->next()) {

    if (frame_idx == 0)
      T_gt0 = frame->T_ab;

    if (frame_idx > num_frames)
      break;
    std::cout << "Frame Idx: " << frame_idx << std::endl;
    frame_idx++;
    lie::SE3 T_BtoA;
    if (frame) {
      ImageFeatures image_feats = frontend.extract_descriptors_(*frame);
      std::vector<cv::KeyPoint> keypoints;
      for (const auto &uv : image_feats.uv) {
        keypoints.emplace_back(uv.x(), uv.y(), 10.0);
      }
      //   std::cout << "Total number of image_features after downsampling: "
      //             << image_feats.uv.size() << std::endl;
      int features_with_valid_depths =
          std::count_if(image_feats.depths.begin(), image_feats.depths.end(),
                        [](const double &d) { return !std::isnan(d); });
      //   std::cout << "Image features wiht valid depths after downsampling: "
      //             << features_with_valid_depths << std::endl
      //             << std::endl
      //             << std::endl;

      std::optional<Estimator::SE3State> state =
          estimator.process_features(image_feats);
      if (!state) {
        continue;
      }
      lie::SE3 T_BtoA = state->x;
      lie::SE3 T_gt = T_gt0.Inverse() * frame->T_ab;
      // TEST STUFF
      //   lie::SE3 dT_est = T1.x.Inverse() * T2.x;
      //   lie::SE3 dT_gt = T_C1toW.x.Inverse() * T_C2toW.x;

      Eigen::Vector3d dxi_phi =
          lie::Minus(T_BtoA.C, T_gt.C, lie::Direction::Right);
      Eigen::Vector3d dr = T_BtoA.p - T_gt.p;
      std::cout << "dxi: " << dxi_phi.transpose() << " dr: " << dr.transpose()
                << std::endl;
      std::cout << "dxi norm: " << dxi_phi.norm() << " dr norm: " << dr.norm()
                << std::endl;

      ASSERT_LE(dxi_phi.norm(), 0.2);
      ASSERT_LE(dr.norm(), 0.2);
    }
  }
  cv::destroyAllWindows();
}
