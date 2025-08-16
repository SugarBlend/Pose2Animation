# cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np

ctypedef np.float32_t FLOAT32_t

def linear_triangulation(np.ndarray[FLOAT32_t, ndim=3] proj_matrices,
                         np.ndarray[FLOAT32_t, ndim=3] points) -> np.ndarray:
    """
    proj_matrices: (n_views, 3, 4)
    points: (n_points, n_views, 2)
    """
    # Transpose points to (n_views, n_points, 2) for broadcasting
    cdef np.ndarray[FLOAT32_t, ndim=3] points_t = points.transpose((1, 0, 2))

    cdef:
        int n_views = points_t.shape[0]
        int n_points = points_t.shape[1]

        np.ndarray[FLOAT32_t, ndim=3] a1 = points_t[:, :, 0:1] * proj_matrices[:, 2:3, :] - proj_matrices[:, 0:1, :]
        np.ndarray[FLOAT32_t, ndim=3] a2 = points_t[:, :, 1:2] * proj_matrices[:, 2:3, :] - proj_matrices[:, 1:2, :]
        np.ndarray[FLOAT32_t, ndim=4] stacked = np.empty((n_views, n_points, 2, 4), dtype=np.float32)
        np.ndarray[FLOAT32_t, ndim=3] A
        np.ndarray[FLOAT32_t, ndim=2] X
        np.ndarray[FLOAT32_t, ndim=3] V

    stacked[:, :, 0, :] = a1
    stacked[:, :, 1, :] = a2

    # Transpose to (n_points, 2, n_views, 4) -> reshape to (n_points, n_views*2, 4)
    A = stacked.transpose((1, 2, 0, 3)).reshape((n_points, n_views * 2, 4))

    U, S, V = np.linalg.svd(A)
    X = V[:, -1]
    X /= X[:, -1][:, None]

    return X[:, :3]
