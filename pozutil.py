"""
POZ Utility Functions and Classes
"""
import numpy as np
import math

RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0


def calc_axes_rotation_mat(roll_phi, pitch_theta, yaw_psi):
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


def calc_xyz_after_rotation_deg(xyz_pos, roll, pitch, yaw):
    """
    Rotates axes by roll-pitch-yaw angles in degrees
    and returns new position with respect to rotated axes.
    Rotate along X, Y, Z in that order to visualize.
    """
    r_rad = roll * DEG2RAD
    p_rad = pitch * DEG2RAD
    y_rad = yaw * DEG2RAD
    ro_mat = calc_axes_rotation_mat(r_rad, p_rad, y_rad)
    return np.dot(ro_mat, np.transpose(xyz_pos))


class Landmark(object):

    def __init__(self, xyz=None, ang_u1max=None, ang_u1min=None, name=""):
        """
        Initializer for Landmark class.
        :param xyz: List with [X, Y, Z] coordinates.
        :param ang_u1max: Adjustment when #1 is RIGHT landmark.
        :param ang_u1min: Adjustment when #1 is LEFT landmark.
        :param name: Optional identifier
        """
        if xyz:
            assert(isinstance(xyz, list))
            self.xyz = np.array(xyz, dtype=float)
        else:
            self.xyz = np.zeros(3, dtype=float)
        self.ang_u1max = ang_u1max
        self.ang_u1min = ang_u1min
        self.name = name
        self.uv = np.array([0., 0.])

    def set_current_uv(self, uv):
        """
        Assign pixel coordinates for latest Landmark sighting.
        :param uv: (U, V) coordinates.
        """
        self.uv = uv

    def calc_world_xz(self, u_var, ang, r):
        """
        Determine world position and pointing direction
        given relative horizontal positions of landmarks
        and previous triangulation result (angle, range).
        :param u_var: Horizontal coordinate of variable landmark
        :param ang: Angle to this landmark
        :param r: Ground range to this landmark
        :return: X, Z in world coordinates
        """
        if self.uv[0] > u_var:
            ang_adj = self.ang_u1max * DEG2RAD - ang
        else:
            ang_adj = self.ang_u1min * DEG2RAD + ang

        # in X,Z plane so need negative sine below
        # to keep azimuth direction consistent
        # (positive azimuth is clockwise)
        world_x = self.xyz[0] + math.cos(ang_adj) * r
        world_z = self.xyz[2] - math.sin(ang_adj) * r
        return world_x, world_z

    def calc_world_azim(self, u_var, ang, rel_azim):
        """
        Convert camera's azimuth to LM to world azimuth.
        Relative azimuth in camera view is also considered.
        :param u_var: U coordinate of variable LM
        :param ang: Angle between camera and LM1-to-LM2 vector
        :param rel_azim: Relative azimuth to LM as seen in image
        :return: World azimuth (radians)
        """
        # there's a 90 degree rotation from camera view to world angle
        if self.uv[0] > u_var:
            offset_rad = self.ang_u1max * DEG2RAD
            world_azim = offset_rad - ang - rel_azim - (np.pi / 2.)
        else:
            offset_rad = self.ang_u1min * DEG2RAD
            world_azim = offset_rad + ang - rel_azim - (np.pi / 2.)
        # clunky way ensure 0 <= world_azim < 360
        if world_azim < 0.:
            world_azim += (2. * np.pi)
        if world_azim < 0.:
            world_azim += (2. * np.pi)
        return world_azim


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
#
# robot camera is always "looking" in its +Z direction
# so its world azimuth is 0 when robot is pointing in +Z direction
# since that is when the two coordinate systems line up
#
# world location is in X,Z plane
# normally the +X axis in X,Z plane would be an angle of 0
# but there is a 90 degree rotation between X,Z and world azimuth


class CameraHelper(object):

    def __init__(self):
        # TODO -- make it accept OpenCV intrinsic camera calib matrix
        # these must be updated prior to triangulation
        self.world_y = 0.
        self.elev = 0.

        # arbitrary test params
        self.w = 640
        self.h = 480
        self.cx = 320
        self.cy = 240
        self.fx = 554  # 60 deg hfov (30.0)
        self.fy = 554  # 46 deg vfov (23.0)

        self.distCoeff = None
        self.camA = np.float32([[self.fx, 0., self.cx],
                                [0., self.fy, self.cy],
                                [0., 0., 1.]])

    def is_visible(self, uv):
        """
        Test if pixel at (u, v) is within valid range.
        :param uv: Numpy array with (u, v) pixel coordinates.
        :return: True if pixel is within image, False otherwise.
        """
        assert(isinstance(uv, np.ndarray))
        result = True
        if int(uv[0]) < 0 or int(uv[0]) >= self.w:
            result = False
        if int(uv[1]) < 0 or int(uv[1]) >= self.h:
            result = False
        return result

    def project_xyz_to_uv(self, xyz):
        """
        Project 3D world point to image plane.
        :param xyz: real world point, shape = (3,)
        :return: Numpy array with (u, v) pixel coordinates.
        """
        assert(isinstance(xyz, np.ndarray))
        pixel_u = self.fx * (xyz[0] / xyz[2]) + self.cx
        pixel_v = self.fy * (xyz[1] / xyz[2]) + self.cy
        return np.array([pixel_u, pixel_v])

    def calc_azim_elev(self, uv):
        """
        Calculate azimuth (radians) and elevation (radians) to image point.
        :param uv: Numpy array with (u, v) pixel coordinates.
        :return: Tuple with azimuth and elevation
        """
        assert(isinstance(uv, np.ndarray))
        ang_azimuth = math.atan((uv[0] - self.cx) / self.fx)
        # need negation here so elevation matches convention listed above
        ang_elevation = math.atan((self.cy - uv[1]) / self.fy)
        return ang_azimuth, ang_elevation

    def calc_rel_xyz_to_pixel(self, known_y, uv, cam_elev):
        """Calculate camera-relative X,Y,Z vector to known landmark in image.
        :param known_y: landmark world Y coord.
        :param uv: Numpy array with Landmark (u, v) pixel coordinates.
        :param cam_elev: camera elevation (radians)
        :return: numpy array [X, Y, Z], shape=(3,)
        """
        assert(isinstance(uv, np.ndarray))
        # use camera params to convert (u, v) to ray
        # u, v might be integers so convert to floats
        # Z coordinate is 1
        ray_x = (float(uv[0]) - self.cx) / self.fx
        ray_y = (float(uv[1]) - self.cy) / self.fy
        ray_cam = np.array([[ray_x], [ray_y], [1.]])

        # rotate ray to undo known camera elevation
        ro_mat_undo_ele = calc_axes_rotation_mat(-cam_elev, 0, 0)
        ray_cam_unrot = np.dot(ro_mat_undo_ele, ray_cam)

        # scale ray based on known height (Y) of landmark
        # this has [X, Y, Z] relative to camera body
        # (can derive angles and ranges from that vector)
        rescale = known_y / ray_cam_unrot[1][0]
        ray_cam_unrot_rescale = np.multiply(ray_cam_unrot, rescale)
        return ray_cam_unrot_rescale.reshape(3,)

    def triangulate_landmarks(self, lm_fix, lm_var):
        """
        Use sightings of a fixed landmark and variable landmark
        to perform triangulation.  Convert angle and range
        from triangulation into world coordinates based
        on fixed landmark's known orientation in world.

        :param lm_fix: Fixed Landmark (#1), known orientation in world.
        :param lm_var: Variable Landmark (#2), orientation may not be known.
        :return: angle, ground range to Landmark 1, world azim for camera
        """
        assert(isinstance(lm_fix, Landmark))
        assert(isinstance(lm_var, Landmark))

        # landmarks can be at different heights
        known_y1 = lm_fix.xyz[1] - self.world_y
        known_y2 = lm_var.xyz[1] - self.world_y

        # find relative vector to landmark 1
        # absolute location of this landmark should be known
        # then calculate ground range
        xyz1 = self.calc_rel_xyz_to_pixel(known_y1, lm_fix.uv, self.elev)
        x1, _, z1 = xyz1
        r1 = math.sqrt(x1 * x1 + z1 * z1)

        # also grab relative azim to landmark 1
        rel_azim = math.atan(x1 / z1)

        # find relative vector to landmark 2
        # this landmark could be point along an edge at unknown position
        # then calculate ground range
        xyz2 = self.calc_rel_xyz_to_pixel(known_y2, lm_var.uv, self.elev)
        x2, _, z2 = xyz2
        r2 = math.sqrt(x2 * x2 + z2 * z2)

        # find vector between landmarks
        # then calculate the ground range between them
        xc, _, zc = xyz2 - xyz1
        c = math.sqrt(xc * xc + zc * zc)

        # all three sides of triangle have been found
        # now use Law of Cosines to calculate angle between the
        # vector to landmark 1 and vector between landmarks
        gamma_cos = (r1 * r1 + c * c - r2 * r2) / (2 * r1 * c)
        angle = math.acos(gamma_cos)

        # landmark has angle offset info
        # which is used to calculate world coords and azim
        u2 = lm_var.uv[0]
        world_azim = lm_fix.calc_world_azim(u2, angle, rel_azim)
        x, z = lm_fix.calc_world_xz(u2, angle, r1)
        return x, z, world_azim


if __name__ == "__main__":
    pass
