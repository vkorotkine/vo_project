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
#include <opencv2/imgproc.hpp>
#include <rerun.hpp>
#include <utility>
// #include <yamlcpp>

using namespace slam_core;
using namespace slam_io;
using namespace slam_types;

// TODO: Need to output feature correspondences between images as well to log
int main() {

  Config cfg = Config(
      EstimatorOptions("baseline", 130, "/home/vio_ws/src/vio_project/output"),
      DataOptions("/home/datasets/tum/rgbd_dataset_freiburg1_xyz", "tum", 0.2),
      FrontendOptions(1000, 3.5, 5, 0.1));

  const auto rec = rerun::RecordingStream("vio");
  std::cout << cfg;
  Dataset::Ptr ds = dataset_factory(cfg.data_options);
  CameraIntrinsics intrinsics = CameraIntrinsics(
      525, 525, 319.5, 239.5, 640, 480, std::array{0., 0., 0., 0., 0.});
  Estimator estimator = Estimator(cfg.estimator_options, intrinsics);
  Frontend frontend = Frontend(cfg.frontend_options);

  std::filesystem::path output_dir(cfg.estimator_options.output_dir);
  std::ofstream out(output_dir / "output.txt");
  rec.save((output_dir / "my_log.rrd").c_str()).exit_on_failure();
  if (!out) {
    std::stringstream err;
    err << "Failed to create output file in" << output_dir;
    throw std::runtime_error(err.str());
  }
  out << " # time (s) x, y, z, qw, qx, qy, qz,  Hamilton convention. Reference "
         "frames: C_CtoW, p_CinW. TUM format. "
      << std::endl;

  int num_frames = 500;
  int frame_idx = 0;

  std::vector<rerun::Vec3D> path;
  while (auto frame = ds->next()) {
    if (frame_idx > num_frames)
      break;
    rec.set_time_sequence("frame", frame_idx);
    std::cout << "Frame Idx: " << frame_idx << std::endl;
    frame_idx++;
    lie::SE3 T_BtoA;
    if (frame) {
      ImageFeatures image_feats = frontend.extract_descriptors_(*frame);
      std::vector<cv::KeyPoint> keypoints;
      for (const auto &uv : image_feats.uv) {
        keypoints.emplace_back(uv.x(), uv.y(), 10.0);
      }
      // cv::drawKeypoints((*frame).rgb, keypoints, temp);
      // cv::imshow("Image", temp);
      // cv::waitKey(1);
      // cv::waitKey(0);
      // image_feats = downsample_to_grid(
      // image_feats, cfg.frontend_options.subsample_grid_height,
      // cfg.frontend_options.subsample_grid_width,
      // cfg.frontend_options.grid_num_points_per_cell);
      // std::cout << "Total number of image_features after downsampling: "
      // << image_feats.uv.size() << std::endl;
      // int features_with_valid_depths =
      //     std::count_if(image_feats.depths.begin(), image_feats.depths.end(),
      //                   [](const double &d) { return !std::isnan(d); });
      // std::cout << "Image features wiht valid depths after downsampling: "
      //           << features_with_valid_depths << std::endl
      //           << std::endl
      //           << std::endl;

      std::optional<Estimator::SE3State> T =
          estimator.process_features(image_feats);
      if (T) {
        out << T->to_line() << std::endl;
        Eigen::Vector3f pf = (T->x.p).cast<float>();
        rerun::Vec3D p{pf(0), pf(1), pf(2)};
        path.push_back(p);
        rec.log("world/trajectory",
                rerun::LineStrips3D(rerun::LineStrip3D(path)));

        const Eigen::Matrix3d &C = T->x.C.Value;
        Eigen::Quaterniond q = Eigen::Quaterniond(C);

        rec.log("world/camera",
                rerun::Transform3D(p, rerun::Quaternion::from_xyzw(
                                          q.x(), q.y(), q.z(), q.w())));
        rec.log("world/camera/",
                rerun::Pinhole::from_focal_length_and_resolution(
                    {static_cast<float>(intrinsics.fx),
                     static_cast<float>(intrinsics.fy)},
                    {static_cast<float>(intrinsics.width),
                     static_cast<float>(intrinsics.height)}));

        // OpenCV is BGR, Rerun RGB.
        // https://docs.opencv.org/4.8.0/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
        cv::Mat rgb;
        const cv::Mat &frame_rgb = frame->rgb;
        cv::cvtColor(frame_rgb, rgb, cv::COLOR_BGR2RGB);
        std::vector<uint8_t> bytes(frame_rgb.data,
                                   frame_rgb.data + frame_rgb.total() *
                                                        frame_rgb.elemSize());
        rec.log("world/camera/image",
                rerun::Image::from_rgb24(
                    bytes, rerun::WidthHeight(uint32_t(frame_rgb.cols),
                                              uint32_t(frame_rgb.rows))));

        // Keypoints in 2D
        std::vector<rerun::Vec2D> kps;
        for (const auto &uv : image_feats.uv) {
          kps.emplace_back(uv(0), uv(1));
        }
        rec.log("world/camera/image", rerun::Points2D(kps).with_radii({3}));

        // Landmarks
        std::vector<rerun::Vec3D> pts;
        std::vector<rerun::Color> colors;
        const std::unordered_map<slam_types::LandmarkId, slam_types::Landmark>
            &landmarks = estimator.get_landmarks();
        //
        for (const auto &[id, lndmrk] : landmarks) {
          pts.emplace_back(lndmrk.position(0), lndmrk.position(1),
                           lndmrk.position(2));
          uint8_t g = std::min<size_t>(lndmrk.observed_in.size() * 60, 255);
          colors.emplace_back(255 - g, g, 0);
        }
        rec.log("world/landmarks",
                rerun::Points3D(pts).with_colors(colors).with_radii({0.015f}));
      }
      keypoints.clear();
      for (const auto &uv : image_feats.uv) {
        keypoints.emplace_back(uv.x(), uv.y(), 10.0);
      }
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