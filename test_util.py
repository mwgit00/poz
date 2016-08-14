
import unittest
import numpy as np
import pozutil as pu
import cv2

from collections import namedtuple
tup_az_el = namedtuple("tup_az_el", "az el")

EPS = 0.01

# LM is short for landmark
#
# 10x16 "room" with 3x9 "alcove"
# with landmarks at some corners
# asterisks are secondary landmarks
# landmark C is in a 135 degree corner (instead of typical 90 degrees)
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
#  |         .     *
#  |               @ F (13,6)
#  |               *
#  |               |
#  *   .           *
#  | A           D |
#  0,0 -*-----*- 13,0 +X

# world location is in X,Z plane
# if looking down at the room
# then positive rotation about Y is clockwise
# the angle from A to B is 0 degrees

# landmarks are higher on the AB side
# ceiling slopes down to CD side
# height is negative to be consistent with right-hand coordinate system
# (+X "cross" +Y points in +Z direction, so +Y points down into floor)
y_ab = -10.
y_cd = -8.
y_offs = -2.

# "fixed" landmarks
mark1 = {"A": pu.Landmark([0., y_ab, 0.], 0., 270.),
         "B": pu.Landmark([0., y_ab, 16.], -270.0, 0.),
         "C": pu.Landmark([8., y_ab + 1.6, 16.], -180., 45.),
         "D": pu.Landmark([13., y_cd, 0.], -90., 180.),
         "E": pu.Landmark([10., y_cd, 9.], -90., 0.),
         "F": pu.Landmark([13., y_cd, 6.], -90., 90.)}

# landmarks that appear to left of fixed landmarks
# (u1 of fixed LM is MAX, or greater than u2 of this LM)
mark2 = {"A": pu.Landmark([2., y_ab + 0.4, 0.]),
         "B": pu.Landmark([0., y_ab, 14.]),
         "C": pu.Landmark([6., y_ab + 1.2, 16.]),
         "D": pu.Landmark([13., y_cd, 2.]),
         "E": pu.Landmark([10., y_cd, 11.]),
         "F": pu.Landmark([13., y_cd + 1., 7.])}

# landmarks that appear to right of fixed landmarks
# (u1 of fixed LM is MIN, or less than u2 of this LM)
mark3 = {"A": pu.Landmark([0., y_ab, 2.]),
         "B": pu.Landmark([2., y_ab + 0.4, 16.]),
         "C": pu.Landmark([10., y_cd, 14.]),
         "D": pu.Landmark([11., y_cd, 0.]),
         "E": pu.Landmark([12., y_cd, 9.]),
         "F": pu.Landmark([13., y_cd + 1., 5.])}

# landmarks that appear below fixed landmarks
markb = {"A": pu.Landmark([0., y_ab - y_offs, 0.], 0., 270.),
         "B": pu.Landmark([0., y_ab - y_offs, 16.], -270.0, 0.),
         "C": pu.Landmark([8., y_ab + 1.6 - y_offs, 16.], -180., 45.),
         "D": pu.Landmark([13., y_cd - y_offs, 0.], -90., 180.),
         "E": pu.Landmark([10., y_cd - y_offs, 9.], -90., 0.),
         "F": pu.Landmark([13., y_cd - y_offs, 6.], -90., 90.)}

# azimuth and elevation of camera so that landmarks
# are visible from (1, 1) at height -3
lm_vis_1_1 = {"A": tup_az_el(225., 70.),
              "B": tup_az_el(0., 30.),
              "C": tup_az_el(30., 0.),
              "D": tup_az_el(90., 30.),
              "E": tup_az_el(45., 15.),
              "F": tup_az_el(60., 15.)}

# azimuth and elevation of camera so that landmarks
# are visible from (7, 6) at height -2
lm_vis_7_6 = {"A": tup_az_el(225., 30.),
              "B": tup_az_el(315., 30.),
              "C": tup_az_el(0., 20.),
              "D": tup_az_el(135., 30.),
              "E": tup_az_el(45., 60.),
              "F": tup_az_el(90., 60.)}


def pnp_test(key, xyz, angs):

    cam = pu.CameraHelper()

    _x, _y, _z = xyz
    _azi, _ele = angs

    cam_xyz = np.float32([_x, _y, _z])

    # world landmark positions
    xyz1_o = mark1[key].xyz
    xyz2_o = mark2[key].xyz
    xyz3_o = mark3[key].xyz
    xyzb_o = markb[key].xyz

    # rotate and offset landmark positions as camera will see them
    xyz1_rot = pu.calc_xyz_after_rotation_deg(xyz1_o - cam_xyz, _ele, _azi, 0)
    xyz2_rot = pu.calc_xyz_after_rotation_deg(xyz2_o - cam_xyz, _ele, _azi, 0)
    xyz3_rot = pu.calc_xyz_after_rotation_deg(xyz3_o - cam_xyz, _ele, _azi, 0)
    xyzb_rot = pu.calc_xyz_after_rotation_deg(xyzb_o - cam_xyz, _ele, _azi, 0)

    # project them to camera plane
    uv1 = cam.project_xyz_to_uv(xyz1_rot)
    uv2 = cam.project_xyz_to_uv(xyz2_rot)
    uv3 = cam.project_xyz_to_uv(xyz3_rot)
    uvb = cam.project_xyz_to_uv(xyzb_rot)

    if cam.is_visible(uv1) and cam.is_visible(uv2) and cam.is_visible(uv3) and cam.is_visible(uvb):

        objectPoints = np.array([xyz1_o, xyz2_o, xyz3_o, xyzb_o])
        imagePoints = np.array([uv1, uv2, uv3, uvb])

        rvecR, tvecR, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, cam.camA, cam.distCoeff)
        if inliers is not None:
            newImagePoints, _ = cv2.projectPoints(objectPoints, rvecR, tvecR, cam.camA, cam.distCoeff)
            # print newImagePoints
            rotM, _ = cv2.Rodrigues(rvecR)
            q = -np.matrix(rotM).T * np.matrix(tvecR)
            print q
        else:
            print "*** PnP failed ***"
    else:
        print "a PnP coord is not visible"


def landmark_test(lm1, lm2,  xyz, angs):

    cam = pu.CameraHelper()

    _x, _y, _z = xyz
    _azi, _ele = angs

    # for the two landmarks:
    # - translate landmark by camera offset
    # - rotate by azimuth and elevation
    # - project into image
    cam_xyz = np.float32([_x, _y, _z])

    # determine pixel location of fixed LM
    xyz1 = lm1.xyz - cam_xyz
    xyz1_rot = pu.calc_xyz_after_rotation_deg(xyz1, _ele, _azi, 0)
    uv1 = cam.project_xyz_to_uv(xyz1_rot)

    # determine pixel location of left/right LM
    xyz2 = lm2.xyz - cam_xyz
    xyz2_rot = pu.calc_xyz_after_rotation_deg(xyz2, _ele, _azi, 0)
    uv2 = cam.project_xyz_to_uv(xyz2_rot)

    if cam.is_visible(uv1) and cam.is_visible(uv2):
        pass
    else:
        print
        print "Image Landmark #1:", uv1
        print "Image Landmark #2:", uv2
        print "At least one landmark is NOT visible!"
        return False, 0., 0., 0.

    # all is well so proceed with test...

    # landmarks have been acquired
    # camera elevation and world Y also need updating
    cam.elev = _ele * pu.DEG2RAD
    cam.world_y = _y

    lm1.set_current_uv(uv1)
    lm2.set_current_uv(uv2)
    world_x, world_z, world_azim = cam.triangulate_landmarks(lm1, lm2)

    # this integer coordinate stuff is disabled for now...
    if False:
        print "Now try with integer pixel coords and known Y coords..."
        lm1.set_current_uv((int(u1 + 0.5), int(v1 + 0.5)))
        lm2.set_current_uv((int(u2 + 0.5), int(v2 + 0.5)))
        print lm1.uv
        print lm2.uv

        world_x, world_z, world_azim = cam.triangulate_landmarks(lm1, lm2)
        print "Robot is at", world_x, world_z, world_azim * pu.RAD2DEG
        print

    return True, world_x, world_z, world_azim * pu.RAD2DEG


def room_test(lm_vis, xyz, lm_name, elev_offset=0.0):
    result = True
    for key in sorted(lm_vis.keys()):
        cam_azim = lm_vis[key].az
        cam_elev = lm_vis[key].el + elev_offset
        angs = [cam_azim, cam_elev]

        markx = eval(lm_name)

        flag, x, z, a = landmark_test(mark1[key], markx[key], xyz, angs)
        if not flag:
            result = False
        if abs(x - xyz[0]) >= EPS:
            result = False
        if abs(z - xyz[2]) >= EPS:
            result = False
        if abs(a - cam_azim) >= EPS and abs(a - 360.0 - cam_azim) >= EPS:
            result = False

    return result


class TestUtil(unittest.TestCase):

    def test_room_x1_z1_y2_lm2_elev00(self):
        # LM name mapped to [world_azim, elev] for visibility at world (1,1)
        # has one case where one landmark is not visible
        xyz = [1., -2., 1.]
        self.assertFalse(room_test(lm_vis_1_1, xyz, "mark2"))

    def test_room_x1_z1_y2_lm3_elev00(self):
        # LM name mapped to [world_azim, elev] for visibility at world (1,1)
        xyz = [1., -2., 1.]
        self.assertTrue(room_test(lm_vis_1_1, xyz, "mark3"))

    def test_room_x1_z1_y2_lm2_elev10(self):
        # LM name mapped to [world_azim, elev] for visibility at world (1,1)
        # camera is at (1, 1) and -2 units high, elevation offset 10 degrees
        xyz = [1., -2., 1.]
        self.assertTrue(room_test(lm_vis_1_1, xyz, "mark2", elev_offset=10.0))

    def test_room_x1_z1_y2_lm3_elev10(self):
        # LM name mapped to [world_azim, elev] for visibility at world (1,1)
        xyz = [1., -2., 1.]
        self.assertTrue(room_test(lm_vis_1_1, xyz, "mark3", elev_offset=10.0))

    def test_room_x1_z1_y3_lm_2elev00(self):
        # LM name mapped to [world_azim, elev] for visibility at world (1,1)
        xyz = [1., -3., 1.]
        self.assertTrue(room_test(lm_vis_1_1, xyz, "mark2"))

    def test_room_x1_z1_y3_lm2_elev00(self):
        # LM name mapped to [world_azim, elev] for visibility at world (1,1)
        xyz = [1., -3., 1.]
        self.assertTrue(room_test(lm_vis_1_1, xyz, "mark3"))

    def test_room_x7_z6_y2_lm2_elev00(self):
        # LM name mapped to [world_azim, elev] for visibility at world (7,6)
        # camera is at (7, 6) and -2 units high
        xyz = [7., -2., 6.]
        self.assertTrue(room_test(lm_vis_7_6, xyz, "mark2"))

    def test_room_x7_z6_y2_lm3_elev00(self):
        # LM name mapped to [world_azim, elev] for visibility at world (7,6)
        # camera is at (7, 6) and -2 units high
        xyz = [7., -2., 6.]
        self.assertTrue(room_test(lm_vis_7_6, xyz, "mark3"))


if __name__ == '__main__':
    unittest.main()
