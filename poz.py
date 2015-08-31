"""POZ Development Application.

"""

import math
import numpy as np
import cv2

import pozutil as pu


def rot_axes_rpy_deg(xyz_pos, r, p, y):
    # rotates axes and returns new position with respect to rotated axes
    # rotate along X, Y, Z
    ro_mat = pu.calc_rot_mat(r * pu.DEG2RAD, p * pu.DEG2RAD, y * pu.DEG2RAD)
    return np.dot(ro_mat, np.transpose(xyz_pos))


def rot_pos_rpy_deg(xyz_pos, r, p, y):
    # rotates vector about axes (flip signs on axis rotation angles)
    # rotate along X, Y, Z
    ro_mat = pu.calc_rot_mat(-r * pu.DEG2RAD, -p * pu.DEG2RAD, -y * pu.DEG2RAD)
    return np.dot(ro_mat, np.transpose(xyz_pos))


def project_points_cv(cam_mat, xyz_arr):
    """Project 3D world point to image plane.
    :param xyz_arr: array of points
    """
    #R = pu.calc_rot_mat(0*pu.DEG2RAD, 0*pu.DEG2RAD, 0.)
    tvec = np.float32([0., 0., 0.])
    #Rrod, _ = cv2.Rodrigues(R)
    rvec = np.zeros((3, 1), dtype=np.float32)
    aaaa, _ = cv2.projectPoints(xyz_arr, rvec, tvec, cam_mat, distCoeffs=None)
    print aaaa
    print


if __name__ == "__main__":
    # TODO -- clean this up

    cam = pu.CameraHelper()

    # "known" test parameters
    landmark_y = -2.0

    # arbitrary for testing
    arb_z = 12.0
    arb_ele = 7.0
    arb_azi = -2.0

    print "Known Y for landmark =", landmark_y
    print

    print "Start with landmark at some XYZ"
    print "Camera elevation = ", 0.0
    print "Camera azimuth = ", 0.0
    xyz = np.array([0., landmark_y, arb_z])
    print(xyz)
    print "Range to landmark = ", np.linalg.norm(xyz)
    u, v = cam.project_xyz_to_uv(np.transpose(xyz))
    print "Landmark image coords = ", (u, v)
    a_el, a_az = cam.calc_elev_azim(u, v)
    print ("ele=", a_el, "azi=", a_az)
    print

    print "Simulate different camera orientation"
    print "Camera elevation = ", arb_ele
    print "Camera azimuth = ", arb_azi
    xyzr = rot_axes_rpy_deg(xyz, arb_ele, arb_azi, 0)
    print(xyzr)
    print "Range to landmark = ", np.linalg.norm(xyzr)
    u, v = cam.project_xyz_to_uv(xyzr)
    print "Landmark NEW image coords = ", (u, v)
    a_el, a_az = cam.calc_elev_azim(u, v)
    print ("ele=", a_el, "azi=", a_az)
    print

    #project_points_cv(cam.camA, np.transpose(xyzr))
    #print

    print "Determine range to landmark using only (u, v)"
    xyz_ray = np.multiply(xyzr, (1. / xyzr[2]))
    print xyz_ray, "Normalize ray to landmark in image"
    xyz_ray_r = rot_axes_rpy_deg(np.transpose(xyz_ray), -arb_ele, 0, 0)
    print xyz_ray_r, "Rotate ray to undo known camera elevation"
    magic = landmark_y / xyz_ray_r[1]
    print magic, "Solve for scale factor to project ray to known Y"
    xyz_ray_adj = np.multiply(xyz_ray_r, magic)
    print xyz_ray_adj
    print "Range to landmark = ", np.linalg.norm(xyz_ray_adj)
    lm_azi = math.atan(xyz_ray_adj[0] / xyz_ray_adj[2]) * pu.RAD2DEG
    print "Azimuth to landmark = ", lm_azi
    print

    print "Using CameraHelper routine..."
    rng, azi = cam.calc_rng_azi_to_landmark(xyz, u, v, arb_ele * pu.DEG2RAD)
    print "Range = ", rng
    print "Azimuth = ", azi * pu.RAD2DEG
    print

    print "Second landmark test..."
    xyz2 = np.array([1.0, landmark_y, arb_z])
    print xyz2
    xyz2r = rot_axes_rpy_deg(xyz2, arb_ele, arb_azi, 0)
    print xyz2r
    u2, v2 = cam.project_xyz_to_uv(xyz2r)
    print "Landmark #2 image coords = ", (u2, v2)
    rng, azi = cam.calc_rng_azi_to_landmark(xyz2, u2, v2, arb_ele * pu.DEG2RAD)
    print "Range = ", rng
    print "Azimuth = ", azi * pu.RAD2DEG

    print "Done."
