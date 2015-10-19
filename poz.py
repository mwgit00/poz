"""
POZ Development Application.
"""

import math
import numpy as np
#import cv2

import pozutil as pu


def rot_axes_rpy_deg(xyz_pos, _roll, _pitch, _yaw):
    # rotates axes by roll-pitch-yaw angles in degrees
    # and returns new position with respect to rotated axes
    # (rotate along X, Y, Z in that order to visualize)
    ro_mat = pu.calc_rot_mat(_roll * pu.DEG2RAD, _pitch * pu.DEG2RAD, _yaw * pu.DEG2RAD)
    return np.dot(ro_mat, np.transpose(xyz_pos))


def perspective_test(_y, _z, _ele, _azi):
    print "--------------------------------------"
    print "Perspective Transform tests"
    print

    cam = pu.CameraHelper()

    # some landmarks in a 3x3 grid pattern
    p0 = np.float32([-1., _y - 1.0, _z])
    p1 = np.float32([0., _y - 1.0, _z])
    p2 = np.float32([1., _y - 1.0, _z])
    p3 = np.float32([-1., _y + 1.0, _z])
    p4 = np.float32([0., _y + 1.0, _z])
    p5 = np.float32([1., _y + 1.0, _z])
    p6 = np.float32([-1., _y, _z])
    p7 = np.float32([0, _y, _z])
    p8 = np.float32([1., _y, _z])

    # 3x3 grid array
    ppp = np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8])
    print "Here are some landmarks in world"
    print ppp

    puv_acc = []
    quv_acc = []
    for vp in ppp:
        # original view of landmarks
        u, v = cam.project_xyz_to_uv(vp)
        puv_acc.append(np.float32([u, v]))
        # rotated view of landmarks
        xyz_r = rot_axes_rpy_deg(vp, _ele, _azi, 0)
        u, v = cam.project_xyz_to_uv(xyz_r)
        quv_acc.append(np.float32([u, v]))
    puv = np.array(puv_acc)
    quv = np.array(quv_acc)

    # 4-pt "diamond" array
    quv4 = np.array([quv[1], quv[4], quv[6], quv[8]])
    puv4 = np.array([puv[1], puv[4], puv[6], puv[8]])

    print
    print "Landmark img coords before rotate:"
    print puv
    print "Landmark img coords after rotate:"
    print quv
    print quv4
    print

    # h, _ = cv2.findHomography(puv, quv)
    # hh = cv2.getPerspectiveTransform(puv4, quv4)
    # print h
    # print hh

    # perspectiveTransform needs an extra dimension
    puv1 = np.expand_dims(puv, axis=0)

    # print "Test perspectiveTransform with findHomography matrix:"
    # xpersp = cv2.perspectiveTransform(puv1, h)
    # print xpersp
    # print "Test perspectiveTransform with getPerspectiveTransform matrix:"
    # xpersp = cv2.perspectiveTransform(puv1, hh)
    # print xpersp
    # print


# 10x16 "room"
# with landmarks at each corner at height -8
# height is negative to be consistent with right-hand coordinate system
# asterisks are secondary landmarks
#
# +Z
# 0,16 -*--*- 10,16
#  | B        C |
#  *            *
#  |            |
#  |            |
#  *            *
#  | A        D |
#  0,0 -*--*- 10,0 +X

# "fixed" landmarks
mark1 = {"A": pu.Landmark([0., -8., 0.], 0, -270),
         "B": pu.Landmark([0., -8., 16.], 270.0, 0),
         "C": pu.Landmark([10., -8., 16.], 180, -90),
         "D": pu.Landmark([10., -8., 0.], 90, -180)}

# landmarks that appear to left of fixed landmarks
mark2 = {"A": pu.Landmark([2., -8., 0.]),
         "B": pu.Landmark([0., -8., 14.]),
         "C": pu.Landmark([8., -8., 16.]),
         "D": pu.Landmark([10., -8., 2.])}

# landmarks that appear to right of fixed landmarks
mark3 = {"A": pu.Landmark([0., -8., 2.]),
         "B": pu.Landmark([2., -8., 16.]),
         "C": pu.Landmark([10., -8., 14.]),
         "D": pu.Landmark([8., -8., 0.])}


def landmark_test(lm1, lm2, _x, _y, _z, _ele, _azi):

    cam = pu.CameraHelper()

    xyz1 = lm1.xyz - np.float32([_x, _y, _z])
    xyz1_r = rot_axes_rpy_deg(xyz1, _ele, _azi, 0)
    u1, v1 = cam.project_xyz_to_uv(xyz1_r)

    xyz2 = lm2.xyz - np.float32([_x, _y, _z])
    xyz2_r = rot_axes_rpy_deg(xyz2, _ele, _azi, 0)
    u2, v2 = cam.project_xyz_to_uv(xyz2_r)

    print "Known Landmark #1:", xyz1
    print "Known Landmark #2:", xyz2
    print "Image Landmark #1:", (u1, v1)
    print "Image Landmark #2:", (u2, v2)
    print

    if cam.is_visible(u1, v1) and cam.is_visible(u2, v2):
        print "Landmarks visible in image!"
        print
    else:
        print "Landmarks not visible"
        return

    print (u1, v1)
    print (u2, v2)
    ang, r = cam.triangulate_with_known_y(xyz1[1], xyz2[1], (u1, v1), (u2, v2), cam_elev)
    print ang * pu.RAD2DEG, r
    world_x, world_z = lm1.calc_world_xz(u1, u2, ang, r)
    print "Robot is at", world_x, world_z
    print

    print "Now try with integer pixel coords and known Y coords..."
    ilm1 = ((int(u1 + 0.5)), (int(v1 + 0.5)))
    ilm2 = ((int(u2 + 0.5)), (int(v2 + 0.5)))
    print ilm1
    print ilm2

    ang, r = cam.triangulate_with_known_y(xyz1[1], xyz2[1], ilm1, ilm2, cam_elev)
    print ang * pu.RAD2DEG, r
    world_x, world_z = lm1.calc_world_xz(u1, u2, ang, r)
    print "Robot is at", world_x, world_z
    print

    print "Done."


if __name__ == "__main__":

    # robot knows this about its camera
    # (arbitrary)
    cam_y = -3.
    cam_elev = 0.  # 7.0

    # robot does not know these
    # (it will have to solve for them)
    cam_x = 1.  # 0.0
    cam_z = 1.  # 12.0
    cam_azi = 30.  # 0,0 for B; 30,0 for C; 120,30 for D; 225,70 for A
    code = "C"

    print "--------------------------------------"
    print "Landmark Test"
    print "Camera Ele =", cam_elev
    print "Camera Azi =", cam_azi
    print "Landmark", code
    print

    landmark_test(mark1[code], mark2[code], cam_x, cam_y, cam_z, cam_elev, cam_azi)

