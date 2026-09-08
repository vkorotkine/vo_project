#include "core/Estimator.hpp"
#include "types/Lie.hpp"
#include "types/Types.hpp"
#include "types/Utils.hpp"
#include <iomanip>
#include <iostream>
#include <map>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <utility>
#include <vector>

namespace slam_core {

slam_types::ImageFeatures
downsample_to_grid(const slam_types::ImageFeatures &feats,
                   int subsample_grid_height, int subsample_grid_width,
                   int num_pts_per_cell) {
  // Bin
  // Keep max_per_cellfeats = feats_downsampled;
  // Rebuild.
  //   opts.init_options.subsample_grid_height

  // Cell to feature index map.
  // cell with idx y, x gets mapped to y*ncol + x.
  std::map<int, std::vector<int>> cells;
  std::vector<int> grid_ybins;
  std::vector<int> grid_xbins;

  for (int idx = 0; idx * subsample_grid_width < feats.img_width; idx++) {
    grid_xbins.emplace_back(idx * subsample_grid_width);
  }
  grid_xbins.emplace_back(feats.img_width);

  for (int idx = 0; idx * subsample_grid_height < feats.img_height; idx++) {
    grid_ybins.emplace_back(idx * subsample_grid_height);
  }
  grid_ybins.emplace_back(feats.img_height);

  int ncols = grid_xbins.size() - 1;
  GridIndexer grid_indexer = GridIndexer(ncols);
  // std::cout << "We be downsampling" << std::endl;
  // std::cout << "Num input feat " << feats.uv.size() << std::endl;

  for (size_t uv_idx = 0; uv_idx < feats.uv.size(); uv_idx++) {
    // std::cout << "Looking at uvidx " << uv_idx << std::endl;
    Eigen::Vector2d uv_vec = feats.uv.at(uv_idx);
    // std::cout << "uv is  " << uv_vec.transpose() << std::endl;

    // int x_idx, y_idx;
    int x_idx = -1;
    int y_idx = -1;

    for (size_t i = 0; i < grid_xbins.size() - 1; i++) {
      if (uv_vec(0) >= grid_xbins.at(i) && uv_vec(0) < grid_xbins.at(i + 1)) {
        // std::cout << "X Bin " << grid_xbins.at(i) << " " <<
        // grid_xbins.at(i +
        // 1)
        // << std::endl;
        x_idx = i;
      }
    }

    for (size_t i = 0; i < grid_ybins.size() - 1; i++) {
      if (uv_vec(1) >= grid_ybins.at(i) && uv_vec(1) < grid_ybins.at(i + 1)) {
        y_idx = i;
        // std::cout << "Y Bin " << grid_ybins.at(i) << " " <<
        // grid_ybins.at(i +
        // 1)
        // << std::endl;
      }
    }
    if (x_idx >= 0 && y_idx >= 0)
      cells[grid_indexer.coord_to_idx(x_idx, y_idx)].push_back(uv_idx);
  }

  // Now we have int->vector of feature indices map
  // where each int (key) corresponds to a bin.

  // Sort s.t. valid depth features are first
  for (auto &[key, feat_idxs] : cells) {
    auto mid =
        std::partition(feat_idxs.begin(), feat_idxs.end(), [&feats](int a) {
          return !std::isnan(feats.depths[a]);
        });
  }

  std::vector<Eigen::Vector2d> uv_downsampled;
  std::vector<double> depths_downsampled;
  cv::Mat descriptors_downsampled;
  std::vector<int> ids_downsampled;
  ;
  // now we go and find the feature that corresponds to each feature index
  // in each bin.
  for (const auto &[key, feat_idxs] : cells) {
    // choose minimum between amount of features we have in the cell, and the
    // maximum that we allow.
    int num_pts_to_add =
        std::min(feat_idxs.size(), std::size_t(num_pts_per_cell));
    for (int i = 0; i < num_pts_to_add; i++) {
      size_t feat_idx = feat_idxs.at(i);
      uv_downsampled.emplace_back(feats.uv.at(feat_idx));
      depths_downsampled.emplace_back(feats.depths.at(feat_idx));
      descriptors_downsampled.push_back(feats.descriptors.row(feat_idx));
      ids_downsampled.emplace_back(feats.ids.at(feat_idx));
    }
  }

  slam_types::ImageFeatures feats_downsampled = slam_types::ImageFeatures{
      feats.stamp,        ids_downsampled,         uv_downsampled,
      depths_downsampled, descriptors_downsampled, feats.img_width,
      feats.img_height};
  return feats_downsampled;
}

void Estimator::initialize_landmarks(
    // This method should get removed by the end.
    // Landmarks are anchored IN THE WORLD FRAME. Make sure this is consistent
    // when changing keyframes.
    // KF should also be identity initialzied.
    slam_types::Keyframe &kf, const slam_types::ImageFeatures &image_feats) {
  for (size_t i = 0; i < image_feats.uv.size(); i++) {
    // i: feature index.
    if (!std::isnan(image_feats.depths[i])) {
      std::vector<slam_types::KeyframeId> keyframe_ids = {kf.id};
      slam_types::LandmarkId lndmrk_id{i};

      Eigen::Vector3d p_LinK =
          backproject_(image_feats.uv[i], image_feats.depths[i]);

      slam_types::Landmark lndmrk(lndmrk_id, p_LinK, keyframe_ids);
      slam_utils::insert_unique(landmarks, lndmrk_id, lndmrk);

      kf.add_landmark_feature_correspondence(lndmrk_id, i);
      latest_landmark_id = lndmrk_id;
    }
  }
}

std::unordered_map<int, int> Estimator::match(cv::Mat cur_descriptors,
                                              cv::Mat kf_descriptors,
                                              bool mutual_consistency,
                                              float lowe_ratio) {

  // for DMatch: each match has query index and train index.
  // the query corresponds to first argument, train corresponds to second
  // argument

  // match both ways
  // Lowe test, make sure first match distance is "much" smaller than next
  // candidate
  std::vector<std::vector<cv::DMatch>> knn_matches_cur2kf;
  matcher.knnMatch(cur_descriptors, kf_descriptors, knn_matches_cur2kf, 2);

  // // So, multiple matches in train may be found for each query.
  std::unordered_map<int, int> cur2kf;
  for (const auto &m : knn_matches_cur2kf) {
    if (m[0].distance < lowe_ratio * m[1].distance) {
      cur2kf[m[0].queryIdx] = m[0].trainIdx;
    }
  }

  if (mutual_consistency) {
    std::vector<std::vector<cv::DMatch>> knn_matches_kf2cur;
    matcher.knnMatch(kf_descriptors, cur_descriptors, knn_matches_kf2cur, 2);

    std::unordered_map<int, int> kf2cur;
    for (const auto &m : knn_matches_kf2cur) {
      if (m[0].distance < lowe_ratio * m[1].distance) {
        kf2cur[m[0].queryIdx] = m[0].trainIdx;
      }
    }

    // Now we need to construct bimap for each match that is present in both.
    std::unordered_map<int, int> cur2kf_unique;
    for (const auto &[cur_idx, kf_idx] : cur2kf) {
      if (kf2cur.count(kf_idx) && (kf2cur.at(kf_idx) == cur_idx))
        cur2kf_unique[cur_idx] = kf_idx;
    }

    return cur2kf_unique;
  }
  return cur2kf;
}

// So. THis thing needs to return a) new landmarks and b) correspondences for
// the old landmarks.
PnPResult
Estimator::global_pose_PnP(const slam_types::ImageFeatures &image_feats,
                           const slam_types::Keyframe &keyframe) {
  // Take in image features observed from current camera frame C.
  // Do PnP w.r.t. world-frame resolved landmarks to get global pose (camera to
  // world).
  // The correspondences are computed by using the previous keyframe, that
  // carries feature/landmark matching information. Landmark World Frame (ID,
  // Pos) -> Keyframe feature ID (+ Descriptor) -> Current image features
  // (descriptor, pixel coord)

  std::unordered_map<int, int> query2train = match(
      image_feats.descriptors, keyframe.features.descriptors,
      opts.matching_opts.mutual_consistency, opts.matching_opts.lowe_ratio);
  // Lowe Test: Check if next match is very similar.
  std::vector<cv::Point3d> pnp_landmarks;
  std::vector<cv::Point2d> pnp_uv;
  std::vector<int> unmatched_features;

  // Populate unmatched features
  for (const auto &[current_frame_feat_idx, kf_feat_idx] : query2train) {
    if (!keyframe.feature_is_landmark(kf_feat_idx)) {
      unmatched_features.emplace_back(current_frame_feat_idx);
    }
  }
  // unmatched features are those that survive Lowe test (so not similar to
  // matched ones) and are also not those that are marked as outliers by the
  // RANSAC.
  auto it =
      std::max_element(unmatched_features.begin(), unmatched_features.end());

  if (it != unmatched_features.end() && *it > image_feats.depths.size()) {
    std::ostringstream os;
    os << "Max unmatched feature index " << *it << " Image depth length "
       << image_feats.depths.size() << " image uv length"
       << image_feats.uv.size();
    throw std::runtime_error(os.str());
  }

  // treat the matched features
  cv::Mat K_cv;
  cv::eigen2cv(intrinsics.K(), K_cv);

  // a) Frame of landmark provided to PnP - world frame.
  // b) Resultant frame transformation vs the one we keep track of - world
  // frame. Ok.
  //  c) Frame of landmarks that are initialized.

  std::unordered_map<int, int> ransac2feat_idx;
  int ransac_idx = 0;
  for (const auto &[current_frame_feat_idx, kf_feat_idx] : query2train) {
    if (!keyframe.feature_is_landmark(kf_feat_idx))
      continue;

    slam_types::LandmarkId lndmrk_id =
        keyframe.feature_to_landmark(kf_feat_idx);
    const Eigen::Vector3d &l = landmarks.at(lndmrk_id).position;
    const Eigen::Vector2d &uv = image_feats.uv.at(current_frame_feat_idx);
    pnp_landmarks.emplace_back(l.x(), l.y(), l.z());
    pnp_uv.emplace_back(uv.x(), uv.y());

    ransac2feat_idx.emplace(ransac_idx, current_frame_feat_idx);
    ransac_idx++;
  }

  cv::Vec3d rvec, tvec;
  std::vector<int> inliers_pnp;
  std::cout << "PnP with " << pnp_uv.size() << " correspondences." << std::endl;
  if (pnp_uv.size() > 6) {
    bool ok = cv::solvePnPRansac(
        pnp_landmarks, pnp_uv, K_cv, cv::noArray(), rvec, tvec, false,
        opts.ransac_pnp.iterationsCount, opts.ransac_pnp.reprojectionError,
        opts.ransac_pnp.confidence, inliers_pnp, cv::SOLVEPNP_ITERATIVE);
    // std::cout << ok << std::endl;
    // std::cout << inliers.size() << std::endl;
    // std::cout << rvec << std::endl;
    // std::cout << tvec << std::endl;
    // Returns world to x-forward frame transformation,
    /// which is same as camera world to z-forward frame transformation

    lie::SO3 C_WtoC =
        lie::SO3::Exp(Eigen::Map<const Eigen::Vector3d>(rvec.val));
    Eigen::Vector3d p_WinC = Eigen::Map<const Eigen::Vector3d>(tvec.val);

    lie::SE3 T_WtoC(C_WtoC, p_WinC);
    lie::SE3 T_CtoW = T_WtoC.Inverse();

    std::cout << "Inliers: " << inliers_pnp.size() << "/" << pnp_uv.size()
              << ", " << inliers_pnp.size() / double(pnp_uv.size()) * 100 << "%"
              << std::endl;
    std::cout << "Inliers PnP size: " << inliers_pnp.size() << std::endl;
    // Now can compute matched correspondences.
    slam_utils::Bimap<size_t, slam_types::LandmarkId> matched_correspondences(
        inliers_pnp.size());
    for (const auto &inlier_pnp : inliers_pnp) {
      // little bit of wizardry
      // ransac indices -> current frame feature indices (since ransac only
      // takes matched features) -> keyframe feature indices -> landmark IDs
      int curframe_feat_idx = ransac2feat_idx.at(inlier_pnp);
      int kf_feat_idx = query2train.at(curframe_feat_idx);

      slam_types::LandmarkId landmark_id =
          keyframe.feature_to_landmark(kf_feat_idx);
      if (!matched_correspondences.value_present(landmark_id))
        matched_correspondences.insert(curframe_feat_idx, landmark_id);
    }

    return PnPResult{T_CtoW, matched_correspondences, unmatched_features};
  } else {
    return PnPResult{std::nullopt, {}, {}};
    ;
  }
}

std::optional<Estimator::SE3State>
Estimator::process_features(const slam_types::ImageFeatures &image_feats) {

  std::cout << "Processing frame with stamp " << std::fixed
            << std::setprecision(6) << image_feats.stamp << std::endl;

  num_frames_since_last_kf++;
  if (!initialized) {
    size_t n_valid = image_feats.num_valid_depths();
    if (image_feats.num_valid_depths() >= opts.init_num_valid_depth_features) {
      initialized = true;
      std::cout << "Initializing with " << n_valid << " features, "
                << opts.init_num_valid_depth_features << " required."
                << std::endl;
      slam_types::KeyframeId key_id{0};
      lie::SE3 T_CtoW;

      slam_types::Keyframe kf =
          slam_types::Keyframe(key_id, T_CtoW, image_feats, n_valid);

      slam_types::ImageFeatures downsampled = downsample_to_grid_(image_feats);

      for (size_t i = 0; i < downsampled.uv.size(); i++) {
        double depth = downsampled.depths.at(i);
        if (std::isnan(depth))
          continue;
        const Eigen::Vector2d &uv = downsampled.uv.at(i);
        slam_types::LandmarkId lndmrk_id =
            slam_types::increment_id(latest_landmark_id);
        latest_landmark_id = lndmrk_id;

        Eigen::Vector3d p_LinC = backproject_(uv, depth);

        // Backproject to obtain p_LinC
        slam_types::Landmark lndmrk(
            lndmrk_id, p_LinC, std::vector<slam_types::KeyframeId>{key_id});

        landmarks.emplace(lndmrk_id, lndmrk);
        // std::cout << "Inserting: Feat ID: " << downsampled.ids.at(i) << ", "
        //           << static_cast<std::uint64_t>(lndmrk_id) << std::endl;
        kf.add_landmark_feature_correspondence(lndmrk_id,
                                               downsampled.ids.at(i));
      }

      std::cout << kf << std::endl;
      keyframes.push_back(kf);
      kf_id_to_index[kf.id] = keyframes.size() - 1;
      return SE3State{image_feats.stamp, lie::SE3{}};
    } else {
      std::cout << "Num valid features: " << n_valid << " less than "
                << opts.init_num_valid_depth_features << " required."
                << std::endl;
      return std::nullopt;
    }
  }

  PnPResult result = global_pose_PnP(image_feats, keyframes.back());
  // Now that its initialized. Check keyframe addition condition.

  if (result.ok()) {
    lie::SE3 T_CtoW = *(result.T);
    int num_correspondences = result.matched_correspondences.size();
    double covis_ratio = static_cast<double>(num_correspondences) /
                         keyframes.back().Feat2Landmark.size();
    bool low_overlap = covis_ratio <= opts.keyframe_opts.covisibility_ratio;

    bool keyframe_add =
        (num_frames_since_last_kf >= opts.keyframe_opts.frames_since_last_kf ||
         (low_overlap &&
          num_correspondences >= opts.keyframe_opts.ransac_inlier_count));
    std::cout << "Conditions : "
              << "Num Frames: " << num_frames_since_last_kf << " vs "
              << opts.keyframe_opts.frames_since_last_kf
              << ", Covisibility Ratio: "
              << result.matched_correspondences.size() << "/"
              << keyframes.back().Feat2Landmark.size() << "=" << covis_ratio
              << " vs " << opts.keyframe_opts.covisibility_ratio << std::endl;

    if (keyframe_add)
      std::cout << "Adding Keyframe.";
    else
      std::cout << "No Keyframe Added.";

    std::cout << std::endl << std::endl;

    if (keyframe_add) {
      num_frames_since_last_kf = 0;

      // this is a little bit annoying because
      // creating kf has to be done in lockstep w/
      // updating keyframe ids.
      slam_types::KeyframeId kf_id{
          static_cast<std::uint64_t>(keyframes.back().id) + 1};

      // old landmarks: do the matched correspondences.

      keyframes.emplace_back(kf_id, T_CtoW, image_feats,
                             result.matched_correspondences);
      kf_id_to_index[kf_id] = keyframes.size() - 1;

      // update the old landmarks that they have
      // been observed in this keyframe

      for (auto &[id, lndmrk] : landmarks)
        lndmrk.num_frames_since_last_obs++;

      for (const auto &[feat_idx, lndmrk_idx] :
           result.matched_correspondences.forward_map()) {
        landmarks.at(lndmrk_idx).observed_in.emplace_back(kf_id);
        landmarks.at(lndmrk_idx).num_frames_since_last_obs = 0;
      }

      // Cull landmarks
      // Have not seen landmark many times +
      // have not seen landmark in a while
      std::vector<slam_types::LandmarkId> lndmrks_to_erase;

      for (auto &[lndmrk_id, lndmrk] : landmarks) {
        bool delete_landmark = lndmrk.observed_in.size() <
                                   opts.lndmrk_culling.min_num_observations &&
                               lndmrk.num_frames_since_last_obs >
                                   opts.lndmrk_culling.num_frames_before_drop;
        if (delete_landmark)
          lndmrks_to_erase.push_back(lndmrk_id);
      }

      for (const auto &lndmrk_id : lndmrks_to_erase) {
        for (const auto &kf_id : landmarks.at(lndmrk_id).observed_in) {
          int kf_index = kf_id_to_index.at(kf_id);
          keyframes.at(kf_index).delete_landmark(lndmrk_id);
        }
        landmarks.erase(lndmrk_id);
      }

      slam_types::ImageFeatures unmatched_all =
          image_feats.get_subset(result.unmatched_features);
      slam_types::ImageFeatures unmatched = downsample_to_grid_(unmatched_all);

      for (size_t i = 0; i < unmatched.uv.size(); i++) {
        double depth = unmatched.depths.at(i);
        if (std::isnan(depth))
          continue;
        const Eigen::Vector2d &uv = unmatched.uv.at(i);
        slam_types::LandmarkId lndmrk_id =
            slam_types::increment_id(latest_landmark_id);
        latest_landmark_id = lndmrk_id;
        slam_types::Landmark lndmrk(lndmrk_id, T_CtoW * backproject_(uv, depth),
                                    std::vector<slam_types::KeyframeId>{kf_id});
        landmarks.emplace(lndmrk_id, lndmrk);
        // std::cout << "Landmark ID: " <<
        // static_cast<std::uint64_t>(lndmrk_id)
        //           << " Feat ID " << unmatched.ids.at(i) << std::endl;
        keyframes.back().add_landmark_feature_correspondence(
            lndmrk_id, unmatched.ids.at(i));
      }

      // add landmarks to keyframe
    }

    return SE3State(image_feats.stamp, T_CtoW);
  } else {
    std::cout << "PnP Failed, returining nullopt" << std::endl;
    return std::nullopt;
  }
}
} // namespace slam_core
