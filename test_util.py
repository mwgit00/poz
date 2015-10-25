
import unittest
import numpy as np
import pozutil as pu


EPS = 0.01

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

# landmarks are higher on the AB side
# ceiling slopes down to CD side
# height is negative to be consistent with right-hand coordinate system
y_ab = -10.
y_cd = -8.

# "fixed" landmarks
mark1 = {"A": pu.Landmark([0., y_ab, 0.], 0., 270.),
         "B": pu.Landmark([0., y_ab, 16.], -270.0, 0.),
         "C": pu.Landmark([8., y_cd, 16.], -180., 45.),
         "D": pu.Landmark([13., y_cd, 0.], -90., 180.),
         "E": pu.Landmark([10., y_cd, 9.], -90., 0.),
         "F": pu.Landmark([13., y_cd, 6.], -90., 90.)}

# landmarks that appear to left of fixed landmarks (u1 is MAX)
mark2 = {"A": pu.Landmark([2., y_ab + 0.4, 0.]),
         "B": pu.Landmark([0., y_ab, 14.]),
         "C": pu.Landmark([6., y_cd, 16.]),
         "D": pu.Landmark([13., y_cd, 2.]),
         "E": pu.Landmark([10., y_cd, 11.]),
         "F": pu.Landmark([13., y_cd, 7.])}

# landmarks that appear to right of fixed landmarks (u1 is MIN)
mark3 = {"A": pu.Landmark([0., y_ab, 2.]),
         "B": pu.Landmark([2., y_ab + 0.4, 16.]),
         "C": pu.Landmark([10., y_cd, 14.]),
         "D": pu.Landmark([11., y_cd, 0.]),
         "E": pu.Landmark([12., y_cd, 9.]),
         "F": pu.Landmark([13., y_cd, 5.])}


lm_vis_1_1 = {"A": [225., 70.],
              "B": [0., 30.],
              "C": [30., 0.],
              "D": [90., 30.],
              "E": [45., 15.],
              "F": [60., 15.]}

lm_vis_7_6 = {"A": [225., 30.],
              "B": [315., 30.],
              "C": [0., 20.],
              "D": [135., 30.],
              "E": [45., 60.],
              "F": [90., 60.]}


def landmark_test(lm1, lm2,  xyz, angs):

    cam = pu.CameraHelper()

    _x, _y, _z = xyz
    _azi, _ele = angs

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
        return False, 0., 0., 0.

    # all is well so proceed with test...

    # landmarks have been acquired
    # camera elevation and world Y also need updating
    cam.elev = _ele * pu.DEG2RAD
    cam.world_y = _y

    lm1.set_current_uv((u1, v1))
    lm2.set_current_uv((u2, v2))
    world_x, world_z, world_azim = cam.triangulate_landmarks(lm1, lm2)

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


def room_test(lm_vis, xyz):
    result = True
    for key in sorted(lm_vis.keys()):
        cam_azim = lm_vis[key][0] + 0.  # change offset for testing
        cam_elev = lm_vis[key][1] + 0.  # cam_elev_offset
        angs = [cam_azim, cam_elev]

        f, x, z, a = landmark_test(mark1[key], mark2[key], xyz, angs)
        if not f:
            result = False
        if abs(x - xyz[0]) >= EPS:
            result = False
        if abs(z - xyz[2]) >= EPS:
            result = False
        if abs(a - cam_azim) >= EPS and abs(a - 360.0 - cam_azim) >= EPS:
            result = False

        f, x, z, a = landmark_test(mark1[key], mark3[key], xyz, angs)
        if not f:
            result = False
        if abs(x - xyz[0]) >= EPS:
            result = False
        if abs(z - xyz[2]) >= EPS:
            result = False
        if abs(a - cam_azim) >= EPS and abs(a - 360.0 - cam_azim) >= EPS:
            result = False

    return result


class TestUtil(unittest.TestCase):

    def test_room_1_1_cam3(self):
        # LM name mapped to [world_azim, elev] for visibility at world (1,1)
        self.assertTrue(room_test(lm_vis_1_1, [1., -3., 1.]))

    def test_room_7_6_cam2(self):
        # LM name mapped to [world_azim, elev] for visibility at world (7,6)
        self.assertTrue(room_test(lm_vis_7_6, [7., -2., 6.]))


if __name__ == '__main__':
    unittest.main()
