from dataclasses import dataclass
from typing import Dict, List
import navlie.lib.states as nv_states
import numpy as np
import pymlg
from evo.core.geometry import umeyama_alignment
from metrics import (
    nv_states2pose_traj,
    sync_nv_states_timestamps,
)


def align_sensors_traj_so3(
    gt_traj: List[np.ndarray],
    sensor_traj: List[np.ndarray],
    C_lb: np.ndarray = None,
) -> List[np.ndarray]:
    """Align a sensor rotation trajectory to a ground-truth rotation trajectory.

    Assumes the two sequences are already synchronised (same number of poses,
    same timestamps).

    Frame convention:

    - GT: body ``b``, world ``a`` → ``C_ab`` provided.
    - Sensor: sensor ``l``, map ``m`` → ``C_ml`` provided.
    - Calibration: ``C_lb`` (from body to sensor).  Identity if not supplied.
    - Unknown alignment: ``C_ma`` solved by aligning the first poses.

    Args:
        gt_traj: Ground-truth attitude sequence as ``C_ab`` rotation matrices.
        sensor_traj: Sensor attitude sequence as ``C_ml`` rotation matrices.
        C_lb: Known extrinsic calibration rotation (body → sensor).
            Defaults to identity.

    Returns:
        Aligned sensor attitudes expressed in the world frame as ``C_ab``.
    """
    # Assume timestamps are the same.
    # Sensor: sensor l, map m. C_ml provided.
    # GT: robot b, world a. C_ab provided.
    # Calibration: C_lb. Provided. From ground truth, to sensor.
    # C_ma required. Arbitrary, product of alignment.
    #   C_ma = C_ml @ C_lb  @ C_ba = C_ml[0] @ C_lb @ C_ab.T[0]. Choose to align beginning.
    C_ma = sensor_traj[0] @ C_lb @ gt_traj[0].T

    # Then, Sensor traj aligned to ground truth:
    #   C_ab_check [k] = C_am @ C_ml[k] @ C_lb = C_ma.T @ C_ml @ C_lb
    C_ab_check_list = [C_ma.T @ C_ml @ C_lb for C_ml in sensor_traj]
    return C_ab_check_list


def align_sensor_traj_to_gt(
    gt_traj: List[nv_states.SE3State],
    sensor_traj: List[nv_states.SE3State],
    T_lb: np.ndarray = None,
    method: str = "align_by_first_pose",
) -> List[nv_states.SE3State]:
    """Align a sensor SE(3) trajectory to a ground-truth trajectory.

    Handles both the unknown map/world frame offset and the known
    sensor–body extrinsic calibration ``T_lb``.

    Frame convention:

    - GT: body ``b``, world ``a`` → ``T_ab`` sequence.
    - Sensor: sensor ``l``, map ``m`` → ``T_ml`` sequence.
    - ``T_lb``: known extrinsic (body → sensor).  Identity if not supplied.
    - ``T_am``: unknown world–map offset, solved by ``method``.

    Full relationship: ``T_ab = T_am @ T_ml @ T_lb``

    Args:
        gt_traj: Ground-truth SE3State sequence (``T_ab``).
        sensor_traj: Sensor SE3State sequence (``T_ml``).
        T_lb: Known extrinsic calibration (4×4 SE(3) matrix, body to sensor).
            Defaults to identity.
        method: World-frame alignment strategy.

            - ``"align_by_first_pose"``: match first poses exactly.
            - ``"umeyama"``: optimal SVD-based alignment of position trajectories
              (sequences are first synchronised).

    Returns:
        Aligned sensor trajectory expressed in the world frame as SE3States.
    """
    # Reference frames:
    # Sensor: l, map: m
    # GT: b, World: a
    # The typical SLAM alignment considers the world frame/map frame arbitrary setting.
    # However, we also need to consider sensor/GT calibration.
    # T_{ab} = T_{am} T_{ml} T_{lb}.
    # T_{am}: Transformation between map and world frame. Arbitary.
    #   Can be determined by aligning first pose, or by umeyama alignment on positions.
    # T_{lb}: Transformation between the two sensors. Not arbitrary.
    # In dataset papers a calibration step has to be done beforehand, such as hand-eye calibration
    # or RPNG's vicon2gt utility. Here, it has to be provided.

    if T_lb is None:
        T_lb = np.eye(4)
    t_gt = np.array([s.stamp for s in gt_traj])
    t_sensor_start = sensor_traj[0].stamp
    idx = np.argmin(np.abs(t_gt - t_sensor_start))

    sensor_traj_calib_aligned = [
        nv_states.SE3State(
            value=T_ml.value @ T_lb,
            stamp=T_ml.stamp,
        )
        for T_ml in sensor_traj
    ]  # T_mb

    if method == "align_by_first_pose":
        T_mb = sensor_traj_calib_aligned[0].value
        T_ab = gt_traj[idx]
        T_am = T_ab.value @ pymlg.SE3.inverse(T_mb)  # Aligns world frames.

    if method == "umeyama":

        gt_traj, sensor_traj_calib_aligned = sync_nv_states_timestamps(
            gt_traj, sensor_traj_calib_aligned
        )
        C_ma, r_m_am, _ = umeyama_alignment(
            nv_states2pose_traj(gt_traj).positions_xyz.T,
            nv_states2pose_traj(sensor_traj_calib_aligned).positions_xyz.T,
            with_scale=False,
        )  # align r_a and r_m
        T_ma = pymlg.SE3.from_components(C_ma, r_m_am)
        T_am = pymlg.SE3.inverse(T_ma)

    sensor_traj_aligned = [
        nv_states.SE3State(
            value=T_am @ T_mb.value,
            stamp=T_mb.stamp,
        )
        for T_mb in sensor_traj_calib_aligned
    ]

    return sensor_traj_aligned
