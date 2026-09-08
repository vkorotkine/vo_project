import argparse
from pymlg import SO3, SE3
import navlie.lib.states as nv_states
from scipy.spatial.transform import Rotation
from pathlib import Path
from typing import List
import navlie.utils as nv_utils
from matplotlib import pyplot as plt
from navlie.lib.states import SE3State, SE23State
from typing import List, Union
from pymlg import SO3, SE3, SE23
import numpy as np
from pathlib import Path
import metrics
import alignment


def main(args):
    state_list_gt: List[nv_states.SE3State] = load_from_tum_format(
        args.gt_file, C_ba=False, jpl=False
    )
    state_list_est: List[nv_states.SE3State] = load_from_tum_format(
        args.traj_file, C_ba=False, jpl=False
    )

    state_list_est = alignment.align_sensor_traj_to_gt(state_list_gt, state_list_est)

    max_t = state_list_est[-1].stamp
    state_list_gt = [s for s in state_list_gt if s.stamp < max_t]
    fig, axs = nv_utils.plot_poses(
        state_list_gt, arrow_length=0.01, step=20, label="GT"
    )
    fig, axs = nv_utils.plot_poses(
        state_list_est, axs, arrow_length=0.01, step=20, label="Est"
    )

    plt.savefig(Path(args.output_dir) / "trajectory_comp.pdf")

    fig, axs = nv_utils.plot_poses(
        state_list_gt, arrow_length=0.01, step=20, plot_2d=True, label="GT"
    )
    fig, axs = nv_utils.plot_poses(
        state_list_est,
        axs,
        arrow_length=0.01,
        step=20,
        plot_2d=True,
        label="Est",
        kwargs_line={"linestyle": "--"},
    )

    plt.savefig(Path(args.output_dir) / "trajectory_comp_2d.pdf")

    ape_rot_error, ape_pos_error = metrics.compute_ape(state_list_gt, state_list_est)

    max_t = state_list_gt[-1].stamp - state_list_gt[0].stamp
    print(f"Max time {max_t:.2f}")
    print(f"APE Rotation Error (deg): {ape_rot_error:.2f}")
    print(f"APE Position Error (m): {ape_pos_error:.2f}")


def load_from_tum_format(
    fpath: str,
    C_ba: bool,
    jpl: bool = False,
) -> List[Union[SE3State, SE23State]]:
    """Load a trajectory from TUM format.

    Lines starting with ``#`` are treated as comments and skipped.
    Lines with 8 fields are parsed as SE(3); lines with 11 fields include
    velocity and are parsed as SE(2,3).

    Args:
        fpath: Path to the TUM trajectory file.
        C_ba: If True, the file stores ``C_ba`` (body-to-world rotation) and
            the attitude is set to ``C_ab = C_ba.T``.  If False, the file
            stores ``C_ab`` directly.
        jpl: If True, negate the quaternion vector part before conversion
            (JPL → Hamilton).  Defaults to False (Hamilton / scipy convention).

    Returns:
        List of SE3State or SE23State objects with ``.attitude = C_ab`` and
        ``.position = r_a_ba``. Note: If quaternion part of the line is all zeros, orientation is set to identity.
    """
    states = []
    with open(fpath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or line.startswith("#") or parts[0] == "#":
                continue

            t = float(parts[0])
            r_a_ba = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            quat = np.array(
                [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
            )

            if jpl:
                quat[0:3] = -quat[0:3]

            if not np.allclose(quat, np.zeros(quat.shape)):
                C = Rotation.from_quat(quat).as_matrix()
            else:
                C = np.eye(3)

            if C_ba:
                C_ab = C.T
            else:
                C_ab = C
            if len(parts) == 8:  # t, r, q, standard tum format
                state = SE3State(value=SE3.from_components(C=C_ab, r=r_a_ba), stamp=t)
            if len(parts) == 11:  # if we have velocity in addition to everything else.
                v_a_ba = np.array([float(parts[8]), float(parts[9]), float(parts[10])])
                state = SE23State(
                    value=SE23.from_components(C=C_ab, r=r_a_ba, v=v_a_ba), stamp=t
                )
            states.append(state)

    return states


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_file",
        # default="/home/vassili/projects/datasets/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt",
        default="/home/datasets/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt",
        help="Ground Truth File",
    )
    parser.add_argument(
        "--traj_file",
        # default="/home/vassili/projects/datasets/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt",
        default="/home/vio_ws/src/vio_project/output/output.txt",
        help="Trajectory file",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/vio_ws/src/vio_project/output",
        help="Output diretory",
    )
    args = parser.parse_args()
    main(args)
