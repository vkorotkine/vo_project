#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <iostream>
#include <types/Lie.hpp>

// For Reprojection Factor.
// Have to cross check the reference frame conventions.
int main() {
  using gtsam::symbol_shorthand::L;
  using gtsam::symbol_shorthand::X;

  auto pixel_noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0);
  gtsam::Vector6 diag_pose_cov;
  diag_pose_cov << 0.1, 0.1, 0.3, 0.1, 0.1, 0.1;
  auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas(diag_pose_cov);
  auto robust = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(1.34), pixel_noise);

  auto K = boost::make_shared<gtsam::Cal3_S2>(500., 500., 0.0, 0.0, 0.0);
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initial;
  initial.insert(L(0), gtsam::Point3(Eigen::Vector3d::Zero()));
  initial.insert(X(5), gtsam::Pose3(Eigen::Vector3d::Zero()));

  lie::SE3 T;
  graph.addPrior(X(5), gtsam::Pose3(gtsam::Rot3(T.C.Value), gtsam::Point3(T.p)),
                 pose_noise);

  std::cout << "beep" << std::endl;
  return 0;
}
