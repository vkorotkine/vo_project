import navlie.lib.states as nv_states
from typing import List, Tuple
import numpy as np
from evo.core.sync import matching_time_indices
from evo.core.trajectory import PoseTrajectory3D
import evo.core.metrics as evo_metrics


def sync_nv_states_timestamps(
    states1: List[nv_states.State], states2: List[nv_states.State], max_diff=0.05
) -> Tuple[List[nv_states.State], List[nv_states.State]]:
    """Synchronise two state sequences by nearest-neighbour timestamp matching.

    Args:
        states1: First state sequence.
        states2: Second state sequence.
        max_diff: Maximum allowable timestamp difference in seconds.
            Pairs whose timestamps differ by more than this are discarded.

    Returns:
        Tuple ``(states1_matched, states2_matched)`` — subsets of the inputs
        with one-to-one timestamp correspondence.
    """
    t1 = np.array([s.stamp for s in states1])
    t2 = np.array([s.stamp for s in states2])
    idx1, idx2 = matching_time_indices(t1, t2, max_diff=max_diff)
    states1_matched = [states1[idx] for idx in idx1]
    states2_matched = [states2[idx] for idx in idx2]
    return states1_matched, states2_matched


def nv_states2pose_traj(states: List[nv_states.SE3State]) -> PoseTrajectory3D:
    """Convert a list of SE3States to an :class:`evo.core.trajectory.PoseTrajectory3D`.

    Args:
        states: Trajectory as navlie SE3State objects.

    Returns:
        ``PoseTrajectory3D`` suitable for use with the ``evo`` metrics library.
    """
    pose_traj = PoseTrajectory3D(
        timestamps=[s.stamp for s in states],
        poses_se3=[s.value for s in states],
    )
    return pose_traj


def compute_ape(
    states1: List[nv_states.SE3State], states2: List[nv_states.SE3State]
) -> Tuple[float, float]:
    """Compute the Root-Mean-Square Absolute Pose Error (APE) between two trajectories.

    The two sequences are first synchronised via
    :func:`sync_nv_states_timestamps` (max 0.05 s gap).

    Args:
        states1: Reference (ground truth) trajectory.
        states2: Estimated trajectory.

    Returns:
        Tuple ``(ape_rot_deg, ape_pos_m)`` — RMS rotation error in degrees and
        RMS translation error in metres.
    """
    ape_rot = evo_metrics.APE(evo_metrics.PoseRelation.rotation_angle_deg)
    ape_pos = evo_metrics.APE(evo_metrics.PoseRelation.translation_part)

    states1_matched, states2_matched = sync_nv_states_timestamps(
        states1, states2, max_diff=0.05
    )

    ape_rot.process_data(
        (
            nv_states2pose_traj(states1_matched),
            nv_states2pose_traj(states2_matched),
        )
    )

    ape_pos.process_data(
        (
            nv_states2pose_traj(states1_matched),
            nv_states2pose_traj(states2_matched),
        )
    )
    N = len(states1_matched)
    ape_rot_error = np.sqrt(1 / N * np.sum(ape_rot.error**2))
    ape_pos_error = np.sqrt(1 / N * np.sum(ape_pos.error**2))
    return ape_rot_error, ape_pos_error
