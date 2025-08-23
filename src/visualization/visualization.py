import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Optional


class MatplotlibVisualizer(object):
    def __init__(self, interval_ms: int = 16) -> None:
        self.interval: int = interval_ms

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.bones_connections: Optional[List[List[int]]] = None
        self.joints_sequence: List[np.ndarray] = []
        self.current_frame: int = 0
        self.anim: Optional[FuncAnimation] = None
        self.ax.view_init(elev=20, azim=-40)
        plt.tight_layout()

    def load_sequence(self, joints_sequence: List[np.ndarray]):
        self.joints_sequence = joints_sequence
        if joints_sequence:
            self.bones_connections = get_body_lines(joints_sequence[0].shape[0])

    def _set_labels(self) -> None:
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.set_zlabel("Z (meters)")
        self.ax.set_title("3D Human Pose")

    def _draw(self, idx: int) -> None:
        if idx >= len(self.joints_sequence):
            return

        joints_3d = self.joints_sequence[idx]

        elev = self.ax.elev
        azim = self.ax.azim
        self.ax.cla()
        self.ax.view_init(elev=elev, azim=azim)

        coords = np.split(joints_3d, 3, axis=1)
        self.ax.scatter(*coords, c="#000000", marker="*", s=2, label="Keypoints")

        if self.bones_connections:
            coords = joints_3d[:, :3]
            for idx_start, idx_end in self.bones_connections:
                if idx_start < len(coords) and idx_end < len(coords):
                    line = coords[[idx_start, idx_end]].T
                    self.ax.plot(*line, c="#808080", linewidth=2)

        self._set_labels()
        self.ax.legend()

        if joints_3d is not None:
            hip_center = (joints_3d[11] + joints_3d[12]) / 2
            self.set_axes_equal(center=hip_center)
        else:
            self.set_axes_equal()

    def start_animation(self) -> None:
        self.anim = FuncAnimation(
            self.fig, self._draw, frames=len(self.joints_sequence), interval=self.interval, cache_frame_data=False
        )
        plt.show()

    def set_axes_equal(self, center: Optional[np.ndarray] = None, fixed_radius: float = 0.85) -> None:
        if center is not None:
            x_middle, y_middle = center[:2]
        else:
            x_limits = self.ax.get_xlim3d()
            y_limits = self.ax.get_ylim3d()
            x_middle = np.mean(x_limits)
            y_middle = np.mean(y_limits)
        z_limits = self.ax.get_zlim3d()
        self.ax.set_xlim3d([x_middle - fixed_radius, x_middle + fixed_radius])
        self.ax.set_ylim3d([y_middle - fixed_radius, y_middle + fixed_radius])
        self.ax.set_zlim3d(z_limits)


def get_body_lines(num_points: int) -> List[List[int]]:
    if num_points == 17:
        from src.visualization.palettes import  COCO_SKELETON_INFO
        body_lines = [item["link"] for item in COCO_SKELETON_INFO.values()]
    elif num_points == 133:
        from src.visualization.palettes import  COCO_WHOLEBODY_SKELETON_INFO
        body_lines = [item["link"] for item in COCO_WHOLEBODY_SKELETON_INFO.values()]
    elif num_points == 308:
        from src.visualization.palettes import  GOLIATH_SKELETON_INFO
        body_lines = [item["link"] for item in GOLIATH_SKELETON_INFO.values()]
        body_lines.extend([[16, 15], [16, 17], [15, 17], [15, 13], [13, 17], [13, 16]])
        body_lines.extend([[18, 19], [19, 20], [20, 18], [18, 14], [19, 14], [20, 14]])
    else:
        raise Exception(f"The number of points transmitted is currently not implemented: {num_points}. "
                        f"Available: 17, 133, 308")
    return body_lines
