"""POZ Development Application.

"""

import math
import numpy as np
import cv2

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

    h, _ = cv2.findHomography(puv, quv)
    hh = cv2.getPerspectiveTransform(puv4, quv4)
    print h
    print hh

    # perspectiveTransform needs an extra dimension
    puv1 = np.expand_dims(puv, axis=0)

    print "Test perspectiveTransform with findHomography matrix:"
    xpersp = cv2.perspectiveTransform(puv1, h)
    print xpersp
    print "Test perspectiveTransform with getPerspectiveTransform matrix:"
    xpersp = cv2.perspectiveTransform(puv1, hh)
    print xpersp
    print


def ground_range(xyz):
    x, _, z = xyz
    return math.sqrt(x * x + z * z)


def triangulate_with_known_y(cam_model, _cam_elev, y1, y2, img_pt1, img_pt2):
    u1, v1 = img_pt1
    u2, v2 = img_pt2
    elev = _cam_elev * pu.DEG2RAD
    xyz1_calc = cam_model.calc_rel_xyz_to_landmark(y1, u1, v1, elev)
    a = ground_range(xyz1_calc)
    xyz2_calc = cam_model.calc_rel_xyz_to_landmark(y2, u2, v2, elev)
    b = ground_range(xyz2_calc)
    c = np.linalg.norm(xyz2_calc - xyz1_calc)
    return a, b, c


mark1 = {"A": [0., -8., 0.],
         "B": [0., -8., 16.],
         "C": [10., -8., 16.],
         "D": [10., -8., 0.]}


mark2 = {"A": [0., -8., 2.],
         "B": [2., -8., 16.],
         "C": [10., -8., 14.],
         "D": [8., -8., 0.]}


def landmark_test(_tag, _x, _y, _z, _ele, _azi):
    print "--------------------------------------"
    print "Landmark Test"
    print "Camera Ele =", _ele
    print "Camera Azi =", _azi
    print

    cam = pu.CameraHelper()

    xyz1 = np.float32(mark1[_tag]) - np.float32([_x, _y, _z])
    xyz1_r = rot_axes_rpy_deg(xyz1, _ele, _azi, 0)
    u1, v1 = cam.project_xyz_to_uv(xyz1_r)

    xyz2 = np.float32(mark2[_tag]) - np.float32([_x, _y, _z])
    xyz2_r = rot_axes_rpy_deg(xyz2, _ele, _azi, 0)
    u2, v2 = cam.project_xyz_to_uv(xyz2_r)

    print "Known Landmark #1:", xyz1
    print "Known Landmark #2:", xyz2
    print "Image Landmark #1:", (u1, v1)
    print "Image Landmark #2:", (u2, v2)
    print

    is_good = True
    if u1 <= 0 or u1 >= 640:
        is_good = False
    if u2 <= 0 or u2 >= 640:
        is_good = False
    if v1 <= 0 or v1 >= 480:
        is_good = False
    if v2 <= 0 or v2 >= 480:
        is_good = False
    if is_good:
        print "Landmarks visible in image!"
        print
    else:
        print "Landmarks not visible"
        return

    ground1, ground2, cc = triangulate_with_known_y(cam, cam_elev, xyz1[1],
                                                    xyz2[1], (u1, v1), (u2, v2))
    print "a = ", ground1
    print "b = ", ground2
    print "c = ", cc
    print

    print "Find angles with three known sides (Law of Cosines):"
    xx0 = pu.triangulate_calc_gamma(ground1, cc, ground2)
    print xx0 * pu.RAD2DEG, math.cos(xx0) * ground1, math.sin(xx0) * ground1
    xx1 = pu.triangulate_calc_gamma(ground2, cc, ground1) * pu.RAD2DEG
    print xx1
    xx2 = pu.triangulate_calc_gamma(ground1, ground2, cc) * pu.RAD2DEG
    print xx2

    print "sum = ", (xx0 * pu.RAD2DEG) + xx1 + xx2
    print

    print "Now try with integer pixel coords and known Y coords..."
    ilm1 = ((int(u1 + 0.5)), (int(v1 + 0.5)))
    ilm2 = ((int(u2 + 0.5)), (int(v2 + 0.5)))
    print ilm1
    print ilm2

    ground1, ground2, cc = triangulate_with_known_y(cam, cam_elev, xyz1[1],
                                                    xyz2[1], ilm1, ilm2)
    print "a = ", ground1
    print "b = ", ground2
    print "c = ", cc
    print

    print "Find angles with three known sides (Law of Cosines):"
    xx0 = pu.triangulate_calc_gamma(ground1, cc, ground2)
    print xx0 * pu.RAD2DEG, math.cos(xx0) * ground1, math.sin(xx0) * ground1
    xx1 = pu.triangulate_calc_gamma(ground2, cc, ground1) * pu.RAD2DEG
    print xx1
    xx2 = pu.triangulate_calc_gamma(ground1, ground2, cc) * pu.RAD2DEG
    print xx2
    print "sum = ", (xx0 * pu.RAD2DEG) + xx1 + xx2
    print

    print "Done."


if __name__ == "__main__":

    # robot knows this about its camera
    # (arbitrary)
    cam_y = -3.
    cam_elev = 0  # 7.0

    # robot does not know these
    # (will solve for them)
    cam_x = 1.  # 0.0
    cam_z = 1.  # 12.0
    cam_azi = 0  # 0,0 for B, 30.0 for C, 120,30 for D

    landmark_test("B", cam_x, cam_y, cam_z, cam_elev, cam_azi)
