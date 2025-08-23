# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
import numpy.linalg as la
from scipy.optimize import least_squares

ctypedef np.float32_t FLOAT32
ctypedef np.uint8_t UINT8


def adaptive_linear_triangulation(
    np.ndarray[FLOAT32, ndim=3] proj_matrices,
    np.ndarray[FLOAT32, ndim=3] points,
    float score_threshold=0.85,
    bint use_svd=True
    ):

    cdef int n_joints = points.shape[0]
    cdef int n_cams = points.shape[1]

    cdef np.ndarray[FLOAT32, ndim=2] result = np.zeros((n_joints, 3), dtype=np.float32)
    cdef np.ndarray[UINT8, ndim=2] masks = np.zeros((n_joints, n_cams), dtype=np.uint8)

    cdef int j, i, count_valid, idx
    cdef np.ndarray[FLOAT32, ndim=2] joint_points
    cdef np.ndarray[FLOAT32, ndim=1] scores
    cdef np.ndarray[UINT8, ndim=1] valid_mask = np.zeros(n_cams, dtype=np.uint8)

    cdef np.ndarray[FLOAT32, ndim=3] selected_proj = None
    cdef np.ndarray[FLOAT32, ndim=2] selected_points = None
    cdef np.ndarray[FLOAT32, ndim=2] a1 = None
    cdef np.ndarray[FLOAT32, ndim=2] a2 = None
    cdef np.ndarray[FLOAT32, ndim=2] A = None
    cdef np.ndarray X = None

    for j in range(n_joints):
        joint_points = points[j]
        scores = joint_points[:, 2]

        count_valid = 0
        for i in range(n_cams):
            if scores[i] > score_threshold:
                valid_mask[i] = 1
                count_valid += 1
            else:
                valid_mask[i] = 0

        if count_valid >= 2:
            selected_proj = np.zeros((count_valid, 3, 4), dtype=np.float32)
            selected_points = np.zeros((count_valid, 2), dtype=np.float32)

            idx = 0
            for i in range(n_cams):
                if valid_mask[i]:
                    selected_proj[idx, :, :] = proj_matrices[i, :, :]
                    selected_points[idx, 0] = joint_points[i, 0]
                    selected_points[idx, 1] = joint_points[i, 1]
                    idx += 1

            for i in range(n_cams):
                masks[j, i] = valid_mask[i]

        else:
            top2_idx = np.argsort(scores)[-2:]
            selected_proj = proj_matrices[top2_idx, :, :]
            selected_points = joint_points[top2_idx, :2]
            for i in range(n_cams):
                masks[j, i] = 0
            masks[j, top2_idx[0]] = 1
            masks[j, top2_idx[1]] = 1

        a1 = selected_points[:, 0:1] * selected_proj[:, 2, :] - selected_proj[:, 0, :]
        a2 = selected_points[:, 1:2] * selected_proj[:, 2, :] - selected_proj[:, 1, :]
        A = np.vstack((a1, a2))

        if use_svd:
            _, _, V = la.svd(A)
            X = V[-1]
            X = X / X[-1]
        else:
            eig_vals, eig_vecs = la.eigh(A.T @ A)
            X = eig_vecs[:, 0]
            X = X / X[-1]

        result[j, :] = X[:3]

    return result, masks


def project_point(P, X):
    X_hom = np.append(X, 1.0)
    x_proj = P @ X_hom
    return x_proj[:2] / x_proj[-1]


def reprojection_error(X, proj_matrices, points_2d):
    errors = [project_point(P, X) - pt for P, pt in zip(proj_matrices, points_2d)]
    return np.concatenate(errors)


def triangulate_nonlinear(np.ndarray[FLOAT32, ndim=3] proj_matrices,
                          np.ndarray[FLOAT32, ndim=3] points_2d) -> np.ndarray:
    cdef int n_joints = points_2d.shape[0]
    cdef np.ndarray[FLOAT32, ndim=2] result = np.zeros((n_joints, 3), dtype=np.float32)

    for j in range(n_joints):
        X_init, masks = adaptive_linear_triangulation(proj_matrices, points_2d[j:j+1])
        mask_flat = np.asarray(masks[0], dtype=bool)
        proj_sel = proj_matrices[mask_flat]
        pts_sel = points_2d[j][mask_flat, :2]

        res = least_squares(
            reprojection_error,
            X_init[0],
            args=(proj_sel, pts_sel),
            method='lm'
        )
        result[j] = res.x.astype(np.float32)

    return result
