import time

import cv2
from enum import Enum
import sys
import math
import numpy as np
import pickle
from pathlib import Path
from omegaconf import OmegaConf, ListConfig
from tqdm import tqdm
from typing import List, Dict, Union, Tuple, Literal

try:
    from adaptive_triangulation import adaptive_linear_triangulation, triangulate_nonlinear
    from linear_triangulation import linear_triangulation
except ImportError:
    from src.triangulation.triangulations import (adaptive_linear_triangulation, triangulate_nonlinear,
                                                  linear_triangulation)
from src.triangulation.triangulations import midpoint_triangulation
from src.visualization.opengl_visualizer import AppWindow, QtWidgets, QtCore
from src.visualization.visualization import MatplotlibVisualizer


class TriangulationType(str, Enum):
    Linear = "linear"
    MidPoint = "midpoint"
    Adaptive = "adaptive"
    Nonlinear = "nonlinear"


def skeletons_from_file(path: str) -> np.ndarray:
    try:
        with Path(path).open(mode="rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return np.array([])
    except pickle.UnpicklingError:
        return np.array([])


def triangulate_joints(
    calibrations: ListConfig,
    cameras_keypoints: List[np.ndarray],
    proj_matrices: np.ndarray,
    method: TriangulationType
) -> np.ndarray:
    _reconstruct_method = {
        TriangulationType.MidPoint: midpoint_triangulation,
        TriangulationType.Linear: linear_triangulation,
        TriangulationType.Nonlinear: triangulate_nonlinear,
        TriangulationType.Adaptive: adaptive_linear_triangulation
    }
    undistorted_joints: List[np.ndarray] = []

    for idx, calibration in enumerate(calibrations):
        K = np.array(calibration.K)
        joints = cameras_keypoints[idx].astype(np.float32)
        joints[:, :2] = cv2.undistortPoints(
            np.ascontiguousarray(joints[:, :2]), K, np.array(calibration.distCoef), P=K
        ).reshape(-1, 2)
        undistorted_joints.append(joints)
    joints = np.stack(undistorted_joints, dtype=np.float32).transpose((1, 0, 2))

    estimation = _reconstruct_method[method](proj_matrices, joints)
    if isinstance(estimation, Tuple):
        return estimation[0]
    return estimation


def load_camera_calibrations(files: Union[List[str], str]) -> ListConfig:
    calibrations: List[Dict[str, np.ndarray]] = []
    for path in files:
        config = OmegaConf.load(path)
        intrinsics = config.intrinsics_w_distortion
        fx, fy = intrinsics.f[0]
        cx, cy = intrinsics.c[0]
        k1, k2, k3 = intrinsics.k[0]
        p1, p2 = intrinsics.p[0]

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])
        dist = np.array([k1, k2, p1, p2, k3])

        R = np.array(config.extrinsics.R)
        C = np.array(config.extrinsics.T).reshape(3, 1)
        t = -R @ C

        calibrations.append({
            "K": K.tolist(),
            "distCoef": dist.tolist(),
            "R": R.tolist(),
            "t": t.tolist()
        })
    return OmegaConf.create(calibrations)


def visualize_3d_reconstruction(
    camera_config_paths: List[str],
    cameras_skeletons: List[np.ndarray],
    method: TriangulationType = TriangulationType.MidPoint,
    fps: float = 1000.,
    visualize_backend: Literal["matplotlib", "opengl"] = "opengl"
) -> None:
    calibration_data = load_camera_calibrations(camera_config_paths)
    frames_number = min([len(item) for item in cameras_skeletons])
    progress_bar = tqdm(total=frames_number, desc="Rendering frames", ncols=70)

    proj_matrices = np.array([
        np.array(calibration.K) @ np.hstack((calibration.R, calibration.t))
        for calibration in calibration_data
    ], dtype=np.float32)

    if visualize_backend == "matplotlib":
        skeletons_3d = []
        for idx in range(frames_number):
            frame_cameras_joints = [cam[idx][0] for cam in cameras_skeletons]
            joints_3d = triangulate_joints(calibration_data, frame_cameras_joints, proj_matrices, method)
            skeletons_3d.append(joints_3d)
            progress_bar.update()

        visualizer = MatplotlibVisualizer(interval_ms=math.ceil(1000 / fps))
        visualizer.load_sequence(skeletons_3d)
        visualizer.start_animation()
    else:
        def update_frame() -> None:
            if progress_bar.n >= frames_number:
                progress_bar.close()
                app.quit()
                return

            frame_cameras_joints = [cam[progress_bar.n][0] for cam in cameras_skeletons]
            joints_3d = triangulate_joints(calibration_data, frame_cameras_joints, proj_matrices, method)
            window.opengl_widget.set_skeleton(joints_3d)
            progress_bar.update()
        app = QtWidgets.QApplication(sys.argv)
        window = AppWindow()
        window.show()
        timer = QtCore.QTimer()
        timer.timeout.connect(update_frame)
        timer.start(math.ceil(1000 / fps))

        sys.exit(app.exec())


if __name__ == "__main__":
    camera_config_paths = [
        # "D:/Datasets/Fit3D/fit3d_train/train/s03/camera_parameters/50591643/dumbbell_overhead_shoulder_press.json",
        # "D:/Datasets/Fit3D/fit3d_train/train/s03/camera_parameters/58860488/dumbbell_overhead_shoulder_press.json",
        # "D:/Datasets/Fit3D/fit3d_train/train/s03/camera_parameters/60457274/dumbbell_overhead_shoulder_press.json",
        # "D:/Datasets/Fit3D/fit3d_train/train/s03/camera_parameters/65906101/dumbbell_overhead_shoulder_press.json"
        "D:/Datasets/Fit3D/fit3d_train/train/s03/camera_parameters/50591643/walk_the_box.json",
        "D:/Datasets/Fit3D/fit3d_train/train/s03/camera_parameters/58860488/walk_the_box.json",
        "D:/Datasets/Fit3D/fit3d_train/train/s03/camera_parameters/60457274/walk_the_box.json",
        "D:/Datasets/Fit3D/fit3d_train/train/s03/camera_parameters/65906101/walk_the_box.json"
    ]
    cameras_skeletons = [
        # "../dataset/dataset_single_man/50591643_skeleton_308.pkl",
        # "../dataset/dataset_single_man/58860488_skeleton_308.pkl",
        # "../dataset/dataset_single_man/60457274_skeleton_308.pkl",
        # "../dataset/dataset_single_man/65906101_skeleton_308.pkl"
        "../dataset/fit3d/walk_the_box/50591643_skeleton_308.pkl",
        "../dataset/fit3d/walk_the_box/58860488_skeleton_308.pkl",
        "../dataset/fit3d/walk_the_box/60457274_skeleton_308.pkl",
        "../dataset/fit3d/walk_the_box/65906101_skeleton_308.pkl"
    ]
    cameras_skeletons = [skeletons_from_file(path) for path in cameras_skeletons]
    visualize_3d_reconstruction(camera_config_paths, cameras_skeletons, visualize_backend="matplotlib")
