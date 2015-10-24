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
        return False

    # all is well so proceed with test...

    # landmarks have been acquired
    # camera elevation and world Y also need updating
    cam.elev = _ele * pu.DEG2RAD
    cam.world_y = _y

    print (u1, v1)
    print (u2, v2)
    lm1.set_current_uv((u1, v1))
    lm2.set_current_uv((u2, v2))
    world_x, world_z = cam.calc_world_xz_azim(lm1, lm2)
    print "Robot is at", world_x, world_z
    print

    print "Now try with integer pixel coords and known Y coords..."
    lm1.set_current_uv((int(u1 + 0.5), int(v1 + 0.5)))
    lm2.set_current_uv((int(u2 + 0.5), int(v2 + 0.5)))
    print lm1.uv
    print lm2.uv

    world_x, world_z = cam.calc_world_xz_azim(lm1, lm2)
    print "Robot is at", world_x, world_z
    print

    print "Done."
    return True


if __name__ == "__main__":

    # robot knows this about its camera
    # (arbitrary)
    cam_y = -3.

    # robot does not know this
    # it will have to solve for it
    cam_x = 1.  # 0.0
    cam_z = 1.  # 12.0

    # TODO -- solve for robot's heading/azimuth somehow

    # landmark code mapped to [azim, elev] for visibility at world (1,1)
    lm_vis = {"A": [225., 70.],
              "B": [0., 0.],
              "C": [30., 0.],
              "D": [90., 30.]}

    print "--------------------------------------"
    print "Landmark Test"
    print

    code = "D"
    cam_azi = lm_vis[code][0]
    cam_elev = lm_vis[code][1]

    print "Camera Ele =", cam_elev
    print "Camera Azi =", cam_azi
    print "Landmark", code

    landmark_test(mark1[code], mark2[code], cam_x, cam_y, cam_z, cam_azi, cam_elev)
