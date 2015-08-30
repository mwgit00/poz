"""POZ Utility Functions and Classes

"""

import numpy as np
import math


RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0


def calc_rot_mat(roll_phi, pitch_theta, yaw_psi):
    """Calculate 3D Euler angle rotation matrix.  Angles in radians."""
    # creates matrix for rotating axes
    # with axis pointing out, positive rotation is clockwise
    # uses right-hand "airplane" conventions:
    # x, forward, roll, phi
    # y, right, pitch, theta
    # z, down, yaw, psi
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


# camera convention
#
# 0 --------- X+
# |           |
# |  (cx,cy)  |
# |           |
# Y+ --------- (w,h)
#
# right-hand rule for Z
# Z- is in (towards screen), Z+ is out (away from screen)
#
# positive elevation is clockwise rotation around X
# positive azimuth is clockwise rotation around Y
# +elevation TO point (u,v) is UP
# +azimuth TO point (u,v) is RIGHT


class CameraHelper(object):

    def __init__(self):
        # TODO -- make it accept OpenCV intrinsic camera calib matrix
        # test params 640 by 480
        self.cx = 320
        self.cy = 240
        self.fx = 554  # 60 deg hfov  (30.0)
        self.fy = 554  # 46 deg vfov (23.0)

    def project_xyz_to_uv(self, xyz_pos):
        """Project 3D world point to image plane."""
        pixel_u = self.fx * (xyz_pos[0][0] / xyz_pos[2][0]) + self.cx
        pixel_v = self.fy * (xyz_pos[1][0] / xyz_pos[2][0]) + self.cy
        return pixel_u, pixel_v

    def calc_elev_azim(self, u, v):
        """Calculate elevation and azimuth.
        :param u: horizontal pixel coordinate
        :param v: vertical pixel coordinate
        """
        ang_elevation = math.atan((self.cy - v) / self.fy)
        ang_elevation *= RAD2DEG
        ang_azimuth = math.atan((u - self.cx) / self.fx)
        ang_azimuth *= RAD2DEG
        return ang_elevation, ang_azimuth

    def calc_range_to_landmark(self, xyz, u, v, elev):
        # landmark is in world at xyz (shape = 1,3)
        # landmark is in image at pixels (u, v)"
        # camera is at known elev (radians)
        # use camera params to convert (u, v) to normalized ray
        ray_x = (u - self.cx) / self.fx
        ray_y = (v - self.cy) / self.fy
        ray_cam = np.array([[ray_x, ray_y, 1.]])
        # rotate ray to undo known camera elevation
        ro_mat_undo_ele = calc_rot_mat(-elev, 0, 0)
        ray_cam_unrot =  np.dot(ro_mat_undo_ele, np.transpose(ray_cam))
        # scale ray based on known height (Y) of landmark
        rescale = xyz[0][1] / ray_cam_unrot[1][0]
        ray_cam_unrot_rescale = np.multiply(ray_cam_unrot, rescale)
        # calculate L2 norm to get distance to landmark
        return np.linalg.norm(ray_cam_unrot_rescale)
