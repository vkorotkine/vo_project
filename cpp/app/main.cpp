#include "core/Config.hpp"
#include "core/Estimator.hpp"
#include "core/Frontend.hpp"
#include "io/Dataset.hpp"
#include "types/Lie.hpp"
#include "types/Types.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <utility>
// #include <yamlcpp>

using namespace slam_core;
using namespace slam_io;
using namespace slam_types;

int main() {

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

  std::filesystem::path output_dir(cfg.estimator_options.output_dir);
  std::ofstream out(output_dir / "output.txt");
  if (!out) {
    std::stringstream err;
    err << "Failed to create output file in" << output_dir;
    throw std::runtime_error(err.str());
  }
  out << " # time (s) x, y, z, qw, qx, qy, qz,  Hamilton convention. Reference "
         "frames: C_CtoW, p_CinW. TUM format. "
      << std::endl;

  int num_frames = 300;
  int frame_idx = 0;

  while (auto frame = ds->next()) {
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
      cv::Mat temp = (*frame).rgb.clone();
      std::cout << *frame;
      // cv::drawKeypoints((*frame).rgb, keypoints, temp);
      // cv::imshow("Image", temp);
      // cv::waitKey(1);
      // cv::waitKey(0);
      // image_feats = downsample_to_grid(
      // image_feats, cfg.frontend_options.subsample_grid_height,
      // cfg.frontend_options.subsample_grid_width,
      // cfg.frontend_options.grid_num_points_per_cell);
      std::cout << "Total number of image_features after downsampling: "
                << image_feats.uv.size() << std::endl;
      int features_with_valid_depths =
          std::count_if(image_feats.depths.begin(), image_feats.depths.end(),
                        [](const double &d) { return !std::isnan(d); });
      std::cout << "Image features wiht valid depths after downsampling: "
                << features_with_valid_depths << std::endl
                << std::endl
                << std::endl;

      std::optional<Estimator::SE3State> T =
          estimator.process_features(image_feats);
      if (T)
        out << T->to_line() << std::endl;

      keypoints.clear();
      for (const auto &uv : image_feats.uv) {
        keypoints.emplace_back(uv.x(), uv.y(), 10.0);
      }
      temp = (*frame).rgb.clone();
      // cv::drawKeypoints((*frame).rgb, keypoints, temp);
      // cv::imshow("Image", temp);

      // cv::waitKey(1);
      // cv::waitKey(0);

      // auto result = estimator.process_frame(image_feats);
      // if (result)
      //   T_BtoA = *result;
    }
  }
  cv::destroyAllWindows();

  return 0;
}