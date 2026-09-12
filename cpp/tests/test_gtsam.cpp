#include <Eigen/Core>
#include <gtest/gtest.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <iostream>
#include <random>
#include <types/Lie.hpp>
#include <types/Types.hpp>
// Lets have a projection camera model
// And generate measurements from it
// Should probably check field of view as well somehow...

TEST(Gtsam, BetweenProjection) {
  using namespace gtsam;
  using symbol_shorthand::L;
  using symbol_shorthand::X;
  using namespace slam_types;
  using Vec6d = Eigen::Matrix<double, 6, 1>;
  using Vec3d = Eigen::Matrix<double, 3, 1>;

  CameraIntrinsics intr = CameraIntrinsics(525, 525, 319.5, 239.5, 640, 480,
                                           std::array{0., 0., 0., 0., 0.});

  boost::shared_ptr<Cal3_S2> Kcal(
      new Cal3_S2(intr.fx, intr.fy, 0., intr.cx, intr.cy));

  Values values;

  values.insert(X(0), Pose3());
  values.insert(X(1), Pose3(Rot3::Ypr(0.1, 0, 0), Point3(1, 0, 0)));
  values.insert(X(2), Pose3(Rot3::Ypr(0.1, 0.2, 0.3), Point3(1, 0.5, 0.2)));

  // noiseless case
  gtsam::NonlinearFactorGraph graph;
  auto measNoise =
      noiseModel::Isotropic::Sigma(2, 1.0); // dimension, stddev. For pixels.
  auto priorNoise = noiseModel::Isotropic::Sigma(6, 0.1); // prior noise
  auto odomNoise = noiseModel::Isotropic::Sigma(6, 0.1);  // prior noise
  auto camNoise = noiseModel::Isotropic::Sigma(2, 0.1);   // prior noise

  auto robust = noiseModel::Robust::Create(
      noiseModel::mEstimator::Huber::Create(1.3), measNoise);

  graph.add(PriorFactor<Pose3>(X(0), Pose3(), priorNoise));
  for (int i = 0; i < 2; i++) {
    graph.add(BetweenFactor<Pose3>(
        X(i), X(i + 1),
        values.at<Pose3>(X(i)).between(values.at<Pose3>(X(i + 1))), odomNoise));
  }
  // relative pose constraints ok
  EXPECT_LE(graph.error(values), 1e-25);

  values.insert(L(0), Point3(1, 1, 1));
  values.insert(L(1), Point3(1, 0, 3));
  values.insert(L(2), Point3(1, 0, 6));

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {

      Point3 p_LinA = values.at<Point3>(L(j));
      Rot3 C_CtoA = values.at<Pose3>(X(i)).rotation();
      Point3 p_CinA = values.at<Pose3>(X(i)).translation();
      Point3 p_LinC = C_CtoA.transpose() * (p_LinA - p_CinA);

      Eigen::Vector2d uv = intr.project_undistorted(p_LinC);
      graph.add(GenericProjectionFactor<Pose3, Point3, Cal3_S2>(
          uv, camNoise, X(i), L(j), Kcal));
    }
  }
  // projection ok
  EXPECT_LE(graph.error(values), 1e-20);

  // add noise, solve, ensure it converges
  std::mt19937 gen(42);
  std::normal_distribution<double> dist(0.0, 0.1);
  std::normal_distribution<double> dist_odom(0.0, 0.01);
  gtsam::NonlinearFactorGraph graph_noise;

  graph_noise.add(PriorFactor<Pose3>(X(0), Pose3(), priorNoise));
  for (int i = 0; i < 2; i++) {
    Eigen::Matrix<double, 6, 1> noise(dist_odom(gen), dist_odom(gen),
                                      dist_odom(gen), dist_odom(gen),
                                      dist_odom(gen), dist_odom(gen));
    graph_noise.add(
        BetweenFactor<Pose3>(X(i), X(i + 1),
                             values.at<Pose3>(X(i))
                                 .between(values.at<Pose3>(X(i + 1)))
                                 .retract(noise),
                             odomNoise));
  }
  // relative pose constraints ok
  EXPECT_LE(graph_noise.error(values), 0.1);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Point2 noise(dist(gen), dist(gen));

      Point3 p_LinA = values.at<Point3>(L(j));
      Rot3 C_CtoA = values.at<Pose3>(X(i)).rotation();
      Point3 p_CinA = values.at<Pose3>(X(i)).translation();
      Point3 p_LinC = C_CtoA.transpose() * (p_LinA - p_CinA);

      Eigen::Vector2d uv = intr.project_undistorted(p_LinC);
      graph_noise.add(GenericProjectionFactor<Pose3, Point3, Cal3_S2>(
          uv + noise, camNoise, X(i), L(j), Kcal));
    }
  }

  LevenbergMarquardtParams p;
  p.setMaxIterations(100);
  p.setRelativeErrorTol(1e-5); // stop if rel error change < tol
  p.setAbsoluteErrorTol(1e-5);
  p.setlambdaInitial(1e-5);
  p.setlambdaUpperBound(1e5);
  p.setVerbosityLM(
      "SUMMARY"); // SILENT|SUMMARY|TERMINATION|LAMBDA|TRYLAMBDA|TRYDELTA
  p.setLinearSolverType("MULTIFRONTAL_CHOLESKY");
  //   p.setOrdering(ordering); // optional custom elimination order

  LevenbergMarquardtOptimizer opt(graph_noise, values, p);
  Values result = opt.optimize();
  EXPECT_LE(graph_noise.error(result), 4);

  for (int i = 0; i < 3; i++) {
    // compute ominus w.r.t. truth
    Vec6d diff =
        values.at<Pose3>(X(i)).localCoordinates(result.at<Pose3>(X(i)));
    EXPECT_LE(diff.norm(), 0.02);
  }
  for (int j = 0; j < 3; j++) {
    Vec3d est = result.at<Point3>(L(j));
    Vec3d true_val = values.at<Point3>(L(j));
    Vec3d diff = est - true_val;
    std::cout << "landmark " << j << std::endl;
    std::cout << "estimated " << est.transpose() << std::endl;
    std::cout << "true " << true_val.transpose() << std::endl;
    EXPECT_LE(
        diff.norm(),
        0.41); // monocular projection factors, so landmark scale will be off
  }
}

TEST(Gtsam, Projection) {
  using namespace gtsam;
  using symbol_shorthand::L;
  using symbol_shorthand::X;
  using namespace slam_types;

  CameraIntrinsics intr = CameraIntrinsics(525, 525, 319.5, 239.5, 640, 480,
                                           std::array{0., 0., 0., 0., 0.});

  boost::shared_ptr<Cal3_S2> Kcal(
      new Cal3_S2(intr.fx, intr.fy, 0., intr.cx, intr.cy));

  Values values;

  values.insert(X(0), Pose3(Rot3::Ypr(0.1, 0.2, 0.3), Point3(1, 0.5, 0.2)));
  values.insert(L(0), Point3(1, 2, 3));

  // noiseless case
  gtsam::NonlinearFactorGraph graph;
  auto priorNoise = noiseModel::Isotropic::Sigma(6, 0.01); // prior noise
  auto camNoise = noiseModel::Isotropic::Sigma(2, 0.01);   // prior noise
  graph.add(PriorFactor<Pose3>(X(0), values.at<Pose3>(X(0)), priorNoise));

  EXPECT_LE(graph.error(values), 1e-25);

  Point3 p_LinA = values.at<Point3>(L(0));
  Rot3 C_CtoA = values.at<Pose3>(X(0)).rotation();
  Point3 p_CinA = values.at<Pose3>(X(0)).translation();
  Point3 p_LinC = C_CtoA.transpose() * (p_LinA - p_CinA);
  //   Point3 p_LinC = values.at<Pose3>(X(0)).inverse() * p_LinC;
  //   std::cout << p_LinA.transpose() << std::endl;

  std::cout << values.at<Pose3>(X(0)).inverse() * p_LinA.transpose()
            << std::endl;

  Eigen::Vector2d uv = intr.project_undistorted(p_LinC);
  graph.add(GenericProjectionFactor<Pose3, Point3, Cal3_S2>(uv, camNoise, X(0),
                                                            L(0), Kcal));
  std::cout << graph.error(values) << std::endl;
  EXPECT_LE(graph.error(values), 1e-20);
}
