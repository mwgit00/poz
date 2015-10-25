"""
POZ Development Application.
"""

import numpy as np
# import cv2

import pozutil as pu


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


# 10x16 "room" with 3x9 "alcove"
# with landmarks at some corners
# asterisks are secondary landmarks
# landmark C is in 135 degree corner
#
# +Z
# 0,16 -*--*- 8,16
#  | B      C  \
#  *            *
#  |            |
#  |            |
#  |            *
#  |            |  E (10,9)
#  |            @--*
#  |               |
#  |         .     |
#  |               |
#  |               |
#  |               |
#  *               *
#  | A           D |
#  0,0 -*-----*- 13,0 +X

# world location is in X,Z plane
# if looking down at the room
# then positive rotation about Y is clockwise


# landmarks are higher on the AB side
# height is negative to be consistent with right-hand coordinate system
y_ab = -10.
y_cd = -8.

# "fixed" landmarks
mark1 = {"A": pu.Landmark([0., y_ab, 0.], 0., 270.),
         "B": pu.Landmark([0., y_ab, 16.], -270.0, 0.),
         "C": pu.Landmark([8., y_cd, 16.], -180., 45.),
         "D": pu.Landmark([13., y_cd, 0.], -90., 180.),
         "E": pu.Landmark([10., y_cd, 9.], -90., 0.)}

# landmarks that appear to left of fixed landmarks (u1 is MAX)
mark2 = {"A": pu.Landmark([2., y_ab, 0.]),
         "B": pu.Landmark([0., y_ab, 14.]),
         "C": pu.Landmark([6., y_cd, 16.]),
         "D": pu.Landmark([13., y_cd, 2.]),
         "E": pu.Landmark([10., y_cd, 11.])}

# landmarks that appear to right of fixed landmarks (u1 is MIN)
mark3 = {"A": pu.Landmark([0., y_ab, 2.]),
         "B": pu.Landmark([2., y_ab, 16.]),
         "C": pu.Landmark([10., y_cd, 14.]),
         "D": pu.Landmark([11., y_cd, 0.]),
         "E": pu.Landmark([12., y_cd, 9.])}


def landmark_test(lm1, lm2, _x, _y, _z, _azi, _ele):
    cam = pu.CameraHelper()

    # for the two landmarks:
    # - translate landmark by camera offset
    # - rotate by azimuth and elevation
    # - project into image

    cam_xyz = np.float32([_x, _y, _z])

    xyz1 = lm1.xyz - cam_xyz
    xyz1_r = pu.calc_xyz_after_rotation_deg(xyz1, _ele, _azi, 0)
    u1, v1 = cam.project_xyz_to_uv(xyz1_r)

    xyz2 = lm2.xyz - cam_xyz
    xyz2_r = pu.calc_xyz_after_rotation_deg(xyz2, _ele, _azi, 0)
    u2, v2 = cam.project_xyz_to_uv(xyz2_r)

    # print "Known Landmark #1:", xyz1
    # print "Known Landmark #2:", xyz2
    if cam.is_visible(u1, v1) and cam.is_visible(u2, v2):
        # print "Both landmarks visible in image!"
        # print
        pass
    else:
        print "Image Landmark #1:", (u1, v1)
        print "Image Landmark #2:", (u2, v2)
        print "At least one landmarks is NOT visible!"
        return False

    # all is well so proceed with test...

    # landmarks have been acquired
    # camera elevation and world Y also need updating
    cam.elev = _ele * pu.DEG2RAD
    cam.world_y = _y

    lm1.set_current_uv((u1, v1))
    lm2.set_current_uv((u2, v2))
    world_x, world_z, world_azim = cam.triangulate_landmarks(lm1, lm2)
    ang = world_azim * pu.RAD2DEG
    print "Robot is at: {:6.3f},{:6.3f},{:20.14f}".format(world_x, world_z, ang)

    if False:
        print "Now try with integer pixel coords and known Y coords..."
        lm1.set_current_uv((int(u1 + 0.5), int(v1 + 0.5)))
        lm2.set_current_uv((int(u2 + 0.5), int(v2 + 0.5)))
        print lm1.uv
        print lm2.uv

        world_x, world_z, world_azim = cam.triangulate_landmarks(lm1, lm2)
        print "Robot is at", world_x, world_z, world_azim * pu.RAD2DEG
        print

    return True


if __name__ == "__main__":

    # robot knows this about its camera
    # (arbitrary)
    cam_y = -3. + 0.  # change offset for testing

    # robot camera is always "looking" in its +Z direction
    # so its world azimuth is 0 when robot is pointing in +Z direction
    # since that is when the two coordinate systems line up

    # LM name mapped to [world_azim, elev] for visibility at world (1,1)
    lm_vis_1_1 = {"A": [225., 70.],
                  "B": [0., 30.],
                  "C": [30., 0.],
                  "D": [90., 30.],
                  "E": [45., 15.]}

    # LM name mapped to [world_azim, elev] for visibility at world (7,6)
    lm_vis_7_6 = {"A": [225., 30.],
                  "B": [315., 30.],
                  "C": [0., 20.],
                  "D": [135., 30.],
                  "E": [45., 60.]}

    tests = [(1., 1., lm_vis_1_1),
             (7., 6., lm_vis_7_6)]

    print "--------------------------------------"
    print "Landmark Test"
    print

    test_index = 0
    vis_map = tests[test_index][2]
    # robot does not know this
    # it will have to solve for it
    cam_x = tests[test_index][0]
    cam_z = tests[test_index][1]
    print cam_x, cam_z
    for key in sorted(vis_map.keys()):
        name = key
        cam_azim = vis_map[name][0] + 0.  # change offset for testing
        cam_elev = vis_map[name][1] + 0.  # change offset for testing
        print "-----------"
        # print "Known Camera Elev =", cam_elev
        print "Landmark {:s}.  Camera Azim = {:8.2f}".format(name, cam_azim)
        landmark_test(mark1[name], mark2[name], cam_x, cam_y, cam_z, cam_azim, cam_elev)
        landmark_test(mark1[name], mark3[name], cam_x, cam_y, cam_z, cam_azim, cam_elev)
