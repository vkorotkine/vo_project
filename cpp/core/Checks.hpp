#pragma once

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <map>
#include <types/Types.hpp>
#include <utility>

namespace slam_core {
void checkGraphValues(const gtsam::NonlinearFactorGraph &graph,
                      const gtsam::Values &values);
void checkLandmark2KeyframeObservations(
    const std::vector<slam_types::Keyframe> &keyframes,
    const slam_types::Landmark &landmark,
    const std::unordered_map<slam_types::KeyframeId, int> &kf_id_to_index);
void checkKeyframe2LandmarkObservations(
    const slam_types::Keyframe &kf,
    const std::unordered_map<slam_types::LandmarkId, slam_types::Landmark>
        &landmarks);
} // namespace slam_core