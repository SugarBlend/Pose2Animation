import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import List, Tuple, Optional

from src.visualization.visualization import get_body_lines


class GLViewWidget(QOpenGLWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self.camera_pos = np.array([0, -1.5, 3.5], dtype=np.float32)
        self.camera_target = np.array([0, 0.2, 0.75], dtype=np.float32)
        self.camera_up = np.array([0, 0, 1], dtype=np.float32)
        self.angle_x: float = np.radians(30)
        self.angle_z: float = np.radians(45)

        self.is_dragging: bool = False
        self.ctrl_pressed: bool = False
        self.last_pos: Optional[QtCore.QPointF] = None

        self.joints_3d = np.zeros((0, 3), dtype=np.float32)
        self.lines: List[List[int]] = []

        self.quad: Optional[OpenGL.GLU.quadrics.LP_GLUQuadric] = None

    def set_skeleton(self, joints_3d: np.ndarray) -> None:
        self.joints_3d = joints_3d.astype(np.float32)
        if not self.lines:
            self.lines = get_body_lines(joints_3d.shape[0])
        self.update()

    @staticmethod
    def draw_axis_arrows(
        origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
        length: float = 0.5,
        radius: float = 0.01,
        cone_height: float = 0.05
    ) -> None:
        ox, oy, oz = origin
        glDisable(GL_LIGHTING)

        glColor3f(1, 0, 0)
        glBegin(GL_LINES)
        glVertex3f(ox, oy, oz)
        glVertex3f(ox + length, oy, oz)
        glEnd()
        glPushMatrix()
        glTranslatef(ox + length, oy, oz)
        glRotatef(90, 0, 1, 0)
        quad = gluNewQuadric()
        gluCylinder(quad, radius * 2, 0.0, cone_height, 12, 1)
        gluDeleteQuadric(quad)
        glPopMatrix()

        glColor3f(0, 1, 0)
        glBegin(GL_LINES)
        glVertex3f(ox, oy, oz)
        glVertex3f(ox, oy + length, oz)
        glEnd()
        glPushMatrix()
        glTranslatef(ox, oy + length, oz)
        glRotatef(-90, 1, 0, 0)
        quad = gluNewQuadric()
        gluCylinder(quad, radius * 2, 0.0, cone_height, 12, 1)
        gluDeleteQuadric(quad)
        glPopMatrix()

        glColor3f(0, 0, 1)
        glBegin(GL_LINES)
        glVertex3f(ox, oy, oz)
        glVertex3f(ox, oy, oz + length)
        glEnd()
        glPushMatrix()
        glTranslatef(ox, oy, oz + length)
        quad = gluNewQuadric()
        gluCylinder(quad, radius * 2, 0.0, cone_height, 12, 1)
        gluDeleteQuadric(quad)
        glPopMatrix()

        glEnable(GL_LIGHTING)

    @staticmethod
    def draw_canvas_background(size: float = 2.0, color: Tuple[float, float, float] = (0.3, 0.5, 0.8)) -> None:
        glDisable(GL_LIGHTING)
        glColor3f(*color)
        glBegin(GL_QUADS)
        glVertex3f(-size, -size, 0)
        glVertex3f(size, -size, 0)
        glVertex3f(size, size, 0)
        glVertex3f(-size, size, 0)
        glEnd()
        glEnable(GL_LIGHTING)

    @staticmethod
    def draw_grid(size: float = 1.0, step: float = 0.1) -> None:
        glDisable(GL_LIGHTING)
        glColor3f(0.5, 0.5, 0.5)
        glBegin(GL_LINES)
        for tic in np.arange(-size, size + step, step):
            glVertex3f(tic, -size, 0)
            glVertex3f(tic, size, 0)
            glVertex3f(-size, tic, 0)
            glVertex3f(size, tic, 0)
        glEnd()
        glEnable(GL_LIGHTING)

    # def draw_skeleton(self) -> None:
    #     if self.joints_3d.shape[0] == 0 or len(self.lines) == 0:
    #         return
    #
    #     glDisable(GL_LIGHTING)
    #     glColor3f(0.65, 0.75, .55)
    #     glLineWidth(1)
    #     glBegin(GL_LINES)
    #     for i1, i2 in self.lines:
    #         if i1 < len(self.joints_3d) and i2 < len(self.joints_3d):
    #             glVertex3fv(self.joints_3d[i1])
    #             glVertex3fv(self.joints_3d[i2])
    #     glEnd()
    #
    #     glPointSize(2)
    #     glColor3f(0.7, 0.7, 0.9)
    #     glBegin(GL_POINTS)
    #     for joint in self.joints_3d:
    #         glVertex3fv(joint)
    #     glEnd()
    #
    #     glEnable(GL_LIGHTING)

    def draw_skeleton(self) -> None:
        if self.joints_3d.shape[0] == 0 or len(self.lines) == 0:
            return

        glEnable(GL_LIGHTING)
        glColor3f(0.65, 0.75, .55)
        for i1, i2 in self.lines:
            if i1 < len(self.joints_3d) and i2 < len(self.joints_3d):
                p1 = self.joints_3d[i1]
                p2 = self.joints_3d[i2]
                self.draw_cylinder_between_points(p1, p2, radius=0.0025, slices=8)

        glColor3f(0.7, 0.7, 0.9)
        for joint in self.joints_3d:
            self.draw_sphere(joint, radius=0.005, slices=8, stacks=8)

    def draw_cylinder_between_points(self, p1: np.ndarray, p2: np.ndarray, radius: float = 0.01, slices: int = 6) -> None:
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length == 0:
            return

        direction /= length

        z_axis = np.array([0, 0, 1])
        rot_axis = np.cross(z_axis, direction)
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)) * 180 / np.pi

        glPushMatrix()
        glTranslatef(*p1)
        if np.linalg.norm(rot_axis) > 1e-6:
            glRotatef(angle, *rot_axis)

        gluCylinder(self.quad, radius, radius, length, slices, 1)

        glPopMatrix()

    def draw_sphere(self, center: np.ndarray, radius: float = 0.02, slices: int = 8, stacks: int = 8) -> None:
        glPushMatrix()
        glTranslatef(*center)
        gluSphere(self.quad, radius, slices, stacks)
        glPopMatrix()

    def initializeGL(self) -> None:
        glClearColor(0.85, 0.85, 0.85, 1.0)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 5.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glEnable(GL_NORMALIZE)

        self.quad = gluNewQuadric()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Control:
            self.ctrl_pressed = True

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Control:
            self.ctrl_pressed = False

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.is_dragging and self.last_pos:
            current_pos = event.position()
            dx = current_pos.x() - self.last_pos.x()
            dy = current_pos.y() - self.last_pos.y()
            self.last_pos = current_pos

            sensitivity = 0.005

            if self.ctrl_pressed:
                forward = self.camera_target - self.camera_pos
                right = np.cross(forward, self.camera_up)
                right /= np.linalg.norm(right)
                up = self.camera_up

                move = -dx * right * sensitivity + dy * up * sensitivity
                self.camera_pos += move
                self.camera_target += move
            else:
                self.angle_z += dx * sensitivity
                self.angle_x += dy * sensitivity
                max_angle_x = np.pi / 2
                self.angle_x = np.clip(self.angle_x, -max_angle_x, max_angle_x)

            self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.last_pos = event.position()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.is_dragging = False

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        r = np.linalg.norm(self.camera_pos - self.camera_target)
        x = self.camera_target[0] + r * np.cos(self.angle_x) * np.sin(self.angle_z)
        y = self.camera_target[1] + r * np.cos(self.angle_x) * np.cos(self.angle_z)
        z = self.camera_target[2] + r * np.sin(self.angle_x)
        cam_pos = np.array([x, y, z], dtype=np.float32)
        gluLookAt(*cam_pos, *self.camera_target, *self.camera_up)

        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 5.0, 1.0])
        self.draw_canvas_background(size=2.0, color=(0.9, 0.9, 0.9))
        self.draw_axis_arrows(origin=np.array([-2.1, -2.1, 0.0]), length=0.4)
        self.draw_grid(size=2.0, step=0.2)
        self.draw_skeleton()

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / h if h != 0 else 1, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y() / 120
        zoom_speed = 0.1

        view_dir = self.camera_pos - self.camera_target
        distance = np.linalg.norm(view_dir)
        if distance <= 0.1 and delta > 0:
            return

        direction = view_dir / distance
        self.camera_pos -= direction * delta * zoom_speed
        self.update()


class AppWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Skeleton viewer")
        self.opengl_widget = GLViewWidget()
        self.setCentralWidget(self.opengl_widget)
