"""POZ Utility Functions and Classes
"""
import numpy as np
import math

RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0


def calc_rot_mat(roll_phi, pitch_theta, yaw_psi):
    """Calculate 3D Euler angle rotation matrix.

    Creates matrix for rotating AXES.
    With axis pointing out, positive rotation is clockwise.
    Uses right-handed "airplane" conventions:
    - x, forward, roll, phi
    - y, right, pitch, theta
    - z, down, yaw, psi

    :param roll_phi: roll angle (radians)
    :param pitch_theta: pitch angle (radians)
    :param yaw_psi: yaw angle (radians)
    """
    rpy = np.eye(3, 3, dtype=np.float32)

    c_r = math.cos(roll_phi)
    s_r = math.sin(roll_phi)
    c_p = math.cos(pitch_theta)
    s_p = math.sin(pitch_theta)
    c_y = math.cos(yaw_psi)
    s_y = math.sin(yaw_psi)

    rpy[0, 0] = c_p * c_y
    rpy[0, 1] = c_p * s_y
    rpy[0, 2] = -s_p

    rpy[1, 0] = (-c_r) * s_y + s_r * s_p * c_y
    rpy[1, 1] = c_r * c_y + s_r * s_p * s_y
    rpy[1, 2] = s_r * c_p

    rpy[2, 0] = s_r * s_y + c_r * s_p * c_y
    rpy[2, 1] = (-s_r) * c_y + c_r * s_p * s_y
    rpy[2, 2] = c_r * c_p

    return rpy


def triangulate_calc_c(a, b, gamma):
    c_squ = (a * a + b * b - (2 * a * b * math.cos(gamma)))
    return math.sqrt(c_squ)


def triangulate_calc_gamma(a, b, c):
    gamma_cos = (a * a + b * b - c * c) / (2 * a * b)
    return math.acos(gamma_cos)


# camera convention
#
# 0 --------- +X
# |           |
# |  (cx,cy)  |
# |           |
# +Y --------- (w,h)
#
# right-hand rule for Z
# -Z is pointing into camera, +Z is pointing away from camera
# +X (fingers) cross +Y (palm) will make +Z (thumb) point away from camera
#
# positive elevation is clockwise rotation around X (axis pointing "out")
# positive azimuth is clockwise rotation around Y (axis pointing "out")
# +elevation TO point (u,v) is UP
# +azimuth TO point (u,v) is RIGHT


class CameraHelper(object):

    def __init__(self):
        # TODO -- make it accept OpenCV intrinsic camera calib matrix
        # test params 640 by 480
        self.cx = 320
        self.cy = 240
        self.fx = 554  # 60 deg hfov (30.0)
        self.fy = 554  # 46 deg vfov (23.0)
        self.camA = np.float32([[self.fx, 0., self.cx],
                                [0., self.fy, self.cy],
                                [0., 0., 1.]])

    def project_xyz_to_uv(self, xyz):
        """Project 3D world point to image plane.
        :param xyz: real world point, shape = (3,)
        """
        pixel_u = self.fx * (xyz[0] / xyz[2]) + self.cx
        pixel_v = self.fy * (xyz[1] / xyz[2]) + self.cy
        return pixel_u, pixel_v

    def calc_azim_elev(self, u, v):
        """Calculate azimuth (radians) and elevation (radians) to image point.
        :param u: horizontal pixel coordinate
        :param v: vertical pixel coordinate
        """
        ang_azimuth = math.atan((u - self.cx) / self.fx)
        # need negation here so elevation matches convention listed above
        ang_elevation = math.atan((self.cy - v) / self.fy)
        return ang_azimuth, ang_elevation,

    def calc_rel_xyz_to_landmark(self, known_y, u, v, cam_elev):
        """Calculate camera-relative X,Y,Z vector to known landmark in image.
        :param known_y: landmark world Y coord.
        :param u: landmark horiz. pixel coord.
        :param v: landmark vert. pixel coord.
        :param cam_elev: camera elevation (radians)
        :return: numpy array [X, Y, Z], shape=(3,)
        """
        # use camera params to convert (u, v) to ray
        # u, v might be integers so convert to floats
        # Z coordinate is 1
        ray_x = (float(u) - self.cx) / self.fx
        ray_y = (float(v) - self.cy) / self.fy
        ray_cam = np.array([[ray_x], [ray_y], [1.]])

        # rotate ray to undo known camera elevation
        ro_mat_undo_ele = calc_rot_mat(-cam_elev, 0, 0)
        ray_cam_unrot = np.dot(ro_mat_undo_ele, ray_cam)

        # scale ray based on known height (Y) of landmark
        # this has [X, Y, Z] relative to camera body
        # (can derive angles and ranges from that vector)
        rescale = known_y / ray_cam_unrot[1][0]
        ray_cam_unrot_rescale = np.multiply(ray_cam_unrot, rescale)
        return ray_cam_unrot_rescale.reshape(3,)
