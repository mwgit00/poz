"""
POZ Development Application.
"""

import numpy as np
# import cv2

import pozutil as pu

import test_util as tpu


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
        xyz_r = pu.calc_xyz_after_rotation_deg(vp, _ele, _azi, 0)
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


if __name__ == "__main__":

    # robot always knows the Y and Elevation of its camera
    # (arbitrary assignments for testing)
    known_cam_y = -3.
    known_cam_el = 0.0

    tests = [(1., 1., tpu.lm_vis_1_1),
             (7., 6., tpu.lm_vis_7_6)]

    print "--------------------------------------"
    print "Landmark Test"
    print

    test_index = 0
    vis_map = tests[test_index][2]

    # robot does not know its (X, Z) position
    # it will have to solve for it
    cam_x = tests[test_index][0]
    cam_z = tests[test_index][1]
    print "Known (X,Z): ", (cam_x, cam_z)

    for key in sorted(vis_map.keys()):
        cam_azim = vis_map[key].az + 0.  # change offset for testing
        cam_elev = vis_map[key].el + known_cam_el
        print "-----------"
        # print "Known Camera Elev =", cam_elev
        xyz = [cam_x, known_cam_y, cam_z]
        angs = [cam_azim, cam_elev]
        print "Landmark {:s}.  Camera Azim = {:8.2f}".format(key, cam_azim)

        lm1 = tpu.mark1[key]
        f, x, z, a = tpu.landmark_test(lm1, tpu.mark2[key], xyz, angs)
        print "Robot is at: {:6.3f},{:6.3f},{:20.14f}".format(x, z, a)
        f, x, z, a = tpu.landmark_test(lm1, tpu.mark3[key], xyz, angs)
        print "Robot is at: {:6.3f},{:6.3f},{:20.14f}".format(x, z, a)
