"""POZ Utility Functions and Classes
"""
import numpy as np
import math

RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0


def calc_rot_mat(roll_phi, pitch_theta, yaw_psi):
    """
    Calculate 3D Euler angle rotation matrix.

    Creates matrix for rotating AXES.
    With axis pointing out, positive rotation is clockwise.
    Uses right-handed "airplane" conventions:
    - x, forward, roll, phi
    - y, right, pitch, theta
    - z, down, yaw, psi

    :param roll_phi: roll angle (radians)
    :param pitch_theta: pitch angle (radians)
    :param yaw_psi: yaw angle (radians)
    :return: numpy array with 3x3 rotation matrix
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


class Landmark(object):

    def __init__(self, _xyz=None, _ang_pos=None, _ang_neg=None, _id=""):
        """
        Initializer for Landmark class.
        :param _xyz: List with [X, Y, Z] coordinates.
        :param _ang_pos: Positive angle adjustment offset.
        :param _ang_neg: Negative angle adjustment offset.
        :param _id: Optional identifier
        """
        if _xyz:
            assert(isinstance(_xyz, list))
            self.xyz = np.array(_xyz, dtype=float)
        else:
            self.xyz = np.zeros(3, dtype=float)
        self.ang_pos = _ang_pos
        self.ang_neg = _ang_neg
        self.id = _id
        self.uv = None

    def calc_world_xz(self, u1, u2, ang, r):
        """
        Determine world position
        given relative horizontal positions of landmarks
        and previous triangulation result (angle, range)
        :param u1: Horizontal coordinate of landmark 1
        :param u2: Horizontal coordinate of landmark 2
        :param ang: Relative azimuth to landmark
        :param r: Ground range to landmark
        :return: (X, Z) in world coordinates
        """
        if u1 < u2:
            ang_adj = self.ang_neg * DEG2RAD - ang
        else:
            ang_adj = self.ang_pos * DEG2RAD + ang
        world_x = self.xyz[0] + math.cos(ang_adj) * r
        world_z = self.xyz[2] + math.sin(ang_adj) * r
        return world_x, world_z



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
        # arbitrary test params
        self.w = 640
        self.h = 480
        self.cx = 320
        self.cy = 240
        self.fx = 554  # 60 deg hfov (30.0)
        self.fy = 554  # 46 deg vfov (23.0)
        self.camA = np.float32([[self.fx, 0., self.cx],
                                [0., self.fy, self.cy],
                                [0., 0., 1.]])

    def is_visible(self, u, v):
        """
        Test if pixel at (u, v) is within valid range.
        :param u: Pixel horizontal coordinate.
        :param v: Pixel vertical coordinate.
        :return: True if pixel is within image, False otherwise.
        """
        result = True
        if int(u) < 0 or int(u) >= self.w:
            result = False
        if int(v) < 0 or int(v) >= self.h:
            result = False
        return result

    def project_xyz_to_uv(self, xyz):
        """
        Project 3D world point to image plane.
        :param xyz: real world point, shape = (3,)
        """
        pixel_u = self.fx * (xyz[0] / xyz[2]) + self.cx
        pixel_v = self.fy * (xyz[1] / xyz[2]) + self.cy
        return pixel_u, pixel_v

    def calc_azim_elev(self, u, v):
        """
        Calculate azimuth (radians) and elevation (radians) to image point.
        :param u: horizontal pixel coordinate
        :param v: vertical pixel coordinate
        :return: Tuple with azimuth and elevation
        """
        ang_azimuth = math.atan((u - self.cx) / self.fx)
        # need negation here so elevation matches convention listed above
        ang_elevation = math.atan((self.cy - v) / self.fy)
        return ang_azimuth, ang_elevation,

    def calc_rel_xyz_to_pixel(self, known_y, u, v, cam_elev):
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

    def triangulate_with_known_y(self, known_y1, known_y2, img_pt1, img_pt2, _cam_elev):
        # assert(isinstance(lm1, Landmark))

        # TODO -- allow case where y1 and y2 differ (not sure how yet)
        u1, v1 = img_pt1
        u2, v2 = img_pt2
        elev = _cam_elev * DEG2RAD

        # find relative vector to landmark 1
        # absolute location of this landmark should be known
        # then calculate ground range
        xyz1 = self.calc_rel_xyz_to_pixel(known_y1, u1, v1, elev)
        x1, _, z1 = xyz1
        r1 = math.sqrt(x1 * x1 + z1 * z1)

        # find relative vector to landmark 2
        # this landmark could be point along an edge at unknown position
        # then calculate ground range
        xyz2 = self.calc_rel_xyz_to_pixel(known_y2, u2, v2, elev)
        x2, _, z2 = xyz2
        r2 = math.sqrt(x2 * x2 + z2 * z2)

        # find vector between landmarks
        # then calculate the ground range between them
        xc, _, zc = xyz2 - xyz1
        c = math.sqrt(xc * xc + zc * zc)

        # all three sides of triangle have been found
        # now calculate angle between
        # vector to landmark 1 and vector between landmarks
        angle = triangulate_calc_gamma(r1, c, r2)

        return angle, r1


if __name__ == "__main__":
    pass
