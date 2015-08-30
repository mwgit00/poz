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


def get_uv_for_rpy_deg(cam_params, xyz_pos, r, p, y):
    xyz_pos_ro = rot_pos_rpy_deg(xyz_pos, r, p, y)
    return cam_params.project_xyz_to_uv(xyz_pos_ro)


def swing_azi(cam_params, xyz_pos, deg1, deg2, step=1):
    for d in range(deg1, deg2 + step, step):
        u, v = get_uv_for_rpy_deg(cam_params, xyz_pos, 0, d, 0)
        print(int(u[0] + 0.5), int(v[0] + 0.5))


def swing_ele(cam_params, xyz_pos, deg1, deg2, step=1):
    for d in range(deg1, deg2 + step, step):
        u, v = get_uv_for_rpy_deg(cam_params, xyz_pos, d, 0, 0)
        print(int(u[0] + 0.5), int(v[0] + 0.5))


def swing(cam_params, xyz_pos, ele, azi):
    ele_step = 1
    if len(ele) == 3:
        ele_step = ele[2]
    azi_step = 1
    if len(azi) == 3:
        azi_step = azi[2]
    for d_ele in range(ele[0], ele[1] + ele_step, ele_step):
        print("ele", d_ele)
        for d_azi in range(azi[0], azi[1] + azi_step, azi_step):
            u, v = get_uv_for_rpy_deg(cam_params, xyz_pos, d_ele, d_azi, 0)
            c_ele, c_azi = cam_params.calc_elev_azim(u, v)
            print(int(u[0]+0.5), int(v[0]+0.5), c_ele, c_azi)


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
    xyz = np.array([[0., landmark_y, arb_z]])
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

    print "Determine range to landmark using (u, v)"
    xyz_ray = np.multiply(xyzr, (1. / xyzr[2]))
    print xyz_ray, "Normalize ray to landmark in image"
    xyz_ray_r = rot_axes_rpy_deg(np.transpose(xyz_ray), -arb_ele, 0, 0)
    print xyz_ray_r, "Rotate ray to undo known camera elevation"
    xyz_ray_rs = np.multiply(xyz_ray_r, abs(1. / xyz_ray_r[2]))
    print xyz_ray_rs, "Normalize ray again"
    magic = landmark_y / xyz_ray_rs[1]
    print magic, "Solve for scale factor to project ray to known Y"
    xyz_ray_adj = np.multiply(xyz_ray_rs, magic)
    print xyz_ray_adj
    print "Range to landmark = ", np.linalg.norm(xyz_ray_adj)
    print

    print "Refined process for determining range to landmark..."
    print "Find landmark in image at pixels (u, v)"
    print u, v
    print "Use camera parameters to convert (u, v) to normalized ray"
    ray_cam = np.array([[(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.]])
    print ray_cam
    print "Rotate ray to undo known camera elevation"
    ro_mat_ele = pu.calc_rot_mat(-arb_ele * pu.DEG2RAD, 0 * pu.DEG2RAD, 0 * pu.DEG2RAD)
    ray_cam_unrot =  np.dot(ro_mat_ele, np.transpose(ray_cam))
    print ray_cam_unrot
    magic = landmark_y / ray_cam_unrot[1][0]
    ray_cam_unrot_magic = np.multiply(ray_cam_unrot, magic)
    print "Scale ray based on known height (Y) of landmark"
    print ray_cam_unrot_magic
    print "Calculate L2 norm to get distance to landmark"
    print np.linalg.norm(ray_cam_unrot_magic)
    print

    print "Using CameraHelper routine..."
    print cam.calc_range_to_landmark(xyz, u, v, arb_ele * pu.DEG2RAD)