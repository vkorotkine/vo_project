#include <algorithm>
#include <core/Checks.hpp>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <map>
#include <sstream>
#include <types/Types.hpp>
#include <utility>

namespace slam_core {
void checkGraphValues(const gtsam::NonlinearFactorGraph &graph,
                      const gtsam::Values &values) {
  for (const auto &f : graph) {
    for (const auto &key : f->keys()) {
      if (!values.exists(key)) {
        std::stringstream is;
        is << "Key not in graph " << gtsam::DefaultKeyFormatter(key);
        throw std::runtime_error(is.str());
      }
    }
  }
  gtsam::KeySet graph_keys = graph.keys();
  for (const auto &key : values.keys()) {
    if (graph_keys.find(key) == graph_keys.end()) {
      std::stringstream is;
      is << "Key in values but not graph " << gtsam::DefaultKeyFormatter(key);
      throw std::runtime_error(is.str());
    }
  }
}
void checkLandmark2KeyframeObservations(
    const std::vector<slam_types::Keyframe> &keyframes,
    const slam_types::Landmark &landmark,
    const std::unordered_map<slam_types::KeyframeId, int> &kf_id_to_index) {

  using namespace slam_types;

  for (const KeyframeId &kf_id : landmark.observed_in) {
    const Keyframe &kf = keyframes.at(kf_id_to_index.at(kf_id));
    if (!kf.Feat2Landmark.reverse_map().count(landmark.id)) {
      std::stringstream is;
      is << "checkLandmark2KeyframeObservations failed " << std::endl;
      is << "Landmark ID " << static_cast<uint64_t>(landmark.id) << " has ";
      is << "Keyframe ID " << static_cast<uint64_t>(kf_id)
         << " in its observed in";
      is << "but is not present in the keyframe Feat2Landmark";
      throw std::runtime_error(is.str());
    }
  }
}

void checkKeyframe2LandmarkObservations(
    const slam_types::Keyframe &kf,
    const std::unordered_map<slam_types::LandmarkId, slam_types::Landmark>
        &landmarks) {
  using namespace slam_types;
  for (const auto &[lndmrk_id, feat_id] : kf.Feat2Landmark.reverse_map()) {
    const Landmark &lndmrk = landmarks.at(lndmrk_id);
    auto it =
        std::find(lndmrk.observed_in.begin(), lndmrk.observed_in.end(), kf.id);

    if (it == lndmrk.observed_in.end()) {
      std::stringstream is;
      is << "checkKeyframe2LandmarkObservations failed " << std::endl;
      is << "Keyframe ID " << static_cast<uint64_t>(kf.id) << " contains ";
      is << "Landmark ID " << static_cast<uint64_t>(lndmrk_id);
      is << "but is not present in the landmark observed_in";
      throw std::runtime_error(is.str());
    }
  }
}
} // namespace slam_core