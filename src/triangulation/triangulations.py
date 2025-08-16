import cv2
import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple


def linear_triangulation(
    proj_matrices: List[np.ndarray],
    points: List[np.ndarray]
) -> np.ndarray:
    proj_matrices = np.asarray(proj_matrices)
    points = np.asarray(points)
    a1 = points[:, :, :1] * proj_matrices[:, 2, :] - proj_matrices[:, 0, :]
    a2 = points[:, :, 1:] * proj_matrices[:, 2, :] - proj_matrices[:, 1, :]
    stacked = np.stack((a1, a2), axis=2)
    A = stacked.reshape(stacked.shape[0], -1, 4)

    U, E, V = np.linalg.svd(A)
    X = V[:, -1]
    X /= X[:, -1][:, None]
    return X[:, :3]


def adaptive_linear_triangulation(
    proj_matrices: np.ndarray,
    points: np.ndarray,
    score_threshold: float = 0.85,
    use_svd: bool = True
) -> Tuple[np.ndarray, List[np.ndarray]]:
    n_joints, n_cams, _ = points.shape
    result = np.zeros((n_joints, 3), dtype=np.float32)
    masks = np.zeros((n_joints, n_cams), dtype=np.uint8)

    for j in range(n_joints):
        joint_points = points[j]
        scores = joint_points[:, 2]
        valid_mask = scores > score_threshold
        if np.sum(valid_mask) >= 2:
            selected_proj = proj_matrices[valid_mask]
            selected_points = joint_points[valid_mask][:, :2]
            masks[j] = valid_mask
        else:
            top2 = np.argsort(scores)[-2:]
            selected_proj = proj_matrices[top2]
            selected_points = joint_points[top2][:, :2]
            masks[j][top2] = 1

        a1 = selected_points[:, :1] * selected_proj[:, 2, :] - selected_proj[:, 0, :]
        a2 = selected_points[:, 1:] * selected_proj[:, 2, :] - selected_proj[:, 1, :]
        A = np.vstack((a1, a2))

        if use_svd:
            _, _, V = np.linalg.svd(A)
            X = V[-1]
            X /= X[-1]
        else:
            eig_vals, eig_vecs = np.linalg.eigh(A.T @ A)
            X = eig_vecs[:, 0]
            X /= X[-1]
        result[j] = X[:3]

    return result, masks


def project_point(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    X_hom = np.append(X, 1)
    x_proj = P @ X_hom
    return x_proj[:2] / x_proj[-1]


def reprojection_error(X: np.ndarray, proj_matrices: List[np.ndarray], points_2d: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([project_point(P, X) - pt for P, pt in zip(proj_matrices, points_2d)])


def triangulate_nonlinear(proj_matrices: List[np.ndarray], points_2d: np.ndarray) -> np.ndarray:
    n_joints = points_2d.shape[0]
    result = np.zeros((n_joints, 3), dtype=np.float32)

    for j in range(n_joints):
        X_init, masks = adaptive_linear_triangulation(np.asarray(proj_matrices), points_2d[j:j+1])
        masks = np.asarray(masks).flatten()
        res = least_squares(
            reprojection_error,
            X_init[0],
            args=(np.asarray(proj_matrices)[masks], points_2d[j][masks, :2]),
            method='lm'
        )
        result[j] = res.x

    return result


def midpoint_triangulation(proj_matrices: np.ndarray, joints: np.ndarray) -> np.ndarray:
    cameras_centers, rays = get_camera_centers_and_rays(proj_matrices, joints)

    N, K, _ = rays.shape
    rays = rays / np.linalg.norm(rays, axis=2, keepdims=True)  # shape: (N, K, 3)

    d_outer = rays[..., :, None] @ rays[..., None, :]  # (N, K, 3, 3)
    I = np.eye(3)[None, None, :, :]  # shape: (1, 1, 3, 3)
    P = I - d_outer  # shape: (N, K, 3, 3)
    A = np.sum(P, axis=0)  # shape: (K, 3, 3)

    # b = sum_i P_i @ C_i
    # camera_centers: (N, 3) → (N, 1, 3, 1)
    C = cameras_centers[:, None, :, None]  # (N, 1, 3, 1)
    b = np.sum(P @ C, axis=0)[..., 0]  # shape: (K, 3)

    return np.linalg.solve(A, b)


def decompose_projection_matrix(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    camera_matrix, rotation_matrix, t_homogeneous, _, _, _, _ = cv2.decomposeProjectionMatrix(P)

    if t_homogeneous.shape[0] == 4:
        camera_center = (t_homogeneous[:3] / t_homogeneous[3]).flatten()
    else:
        raise ValueError("A homogeneous camera center vector (4x1) was expected, but 't' with shape was "
                         f"obtained: {t_homogeneous.shape}")

    return camera_center, rotation_matrix


def get_camera_centers_and_rays(proj_matrices: np.ndarray, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    joints = joints[:, :, :2]
    num_joints, num_views, num_coord = joints.shape
    camera_centers: List[np.ndarray] = []
    rays: List[np.ndarray] = []

    for i, P in enumerate(proj_matrices):
        C, R = decompose_projection_matrix(P)
        camera_centers.append(C)

        x_hom = np.hstack([joints[:, i, :], np.ones((num_joints, 1))])
        M = P[:, :3]
        dirs = np.linalg.inv(M) @ x_hom.T  # (3, K)
        dirs = dirs.T  # (K, 3)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        rays.append(dirs)

    camera_centers: np.ndarray = np.array(camera_centers)
    rays: np.ndarray = np.stack(rays, axis=0)

    return camera_centers, rays
