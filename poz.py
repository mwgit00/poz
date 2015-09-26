"""POZ Development Application.

"""

import math
import numpy as np
import cv2

import pozutil as pu


def rot_axes_rpy_deg(xyz_pos, r, p, y):
    # rotates axes by roll-pitch-yaw angles in degrees
    # and returns new position with respect to rotated axes
    # (rotate along X, Y, Z in that order to visualize)
    ro_mat = pu.calc_rot_mat(r * pu.DEG2RAD, p * pu.DEG2RAD, y * pu.DEG2RAD)
    return np.dot(ro_mat, np.transpose(xyz_pos))


if __name__ == "__main__":

     # TODO -- clean this up
    cam = pu.CameraHelper()

    # "known" test parameters
    landmark_y = -2.0

    # arbitrary for testing
    arb_z = 12.0
    arb_ele = 7.0
    arb_azi = -2.0

    print "--------------------------------------"
    print "Perspective Transform tests"
    print

    # arrange some landmarks in a "diamond" pattern
    p1 = np.float32([1., landmark_y, arb_z])
    p2 = np.float32([-1., landmark_y, arb_z])
    p3 = np.float32([0., landmark_y + 1.0, arb_z])
    p4 = np.float32([0., landmark_y - 1.0, arb_z])
    pp = np.array([p1, p2, p3, p4])
    print "Here are some landmarks in world"
    print pp

    puv_acc = []
    quv_acc = []
    for vp in pp:
        # original view of landmarks
        u, v = cam.project_xyz_to_uv(vp)
        puv_acc.append(np.float32([u, v]))
        # rotated view of landmarks
        xyz_r = rot_axes_rpy_deg(vp, arb_ele, arb_azi, 0)
        u, v = cam.project_xyz_to_uv(xyz_r)
        quv_acc.append(np.float32([u, v]))
    puv = np.array(puv_acc)
    quv = np.array(quv_acc)

    print
    print "Landmark img coords before rotate:"
    print puv
    print "Landmark img coords after rotate:"
    print quv
    print

    H, _ = cv2.findHomography(puv, quv)
    HH = cv2.getPerspectiveTransform(puv, quv)

    # perspectiveTransform needs an extra dimension
    puv1 = np.expand_dims(puv, axis=0)

    print "Test perspectiveTransform with findHomography matrix:"
    xpersp = cv2.perspectiveTransform(puv1, H)
    print xpersp
    print "Test perspectiveTransform with getPerspectiveTransform matrix:"
    xpersp = cv2.perspectiveTransform(puv1, HH)
    print xpersp

    print
    print "--------------------------------------"
    print "Known Y for landmark =", landmark_y
    print
    print "Start with zeroed camera angles and landmark XYZ:"
    print "Camera elevation = ", 0.0
    print "Camera azimuth = ", 0.0
    xyz = np.float32([0., landmark_y, arb_z])
    print(xyz)
    fu, fv = cam.project_xyz_to_uv(np.transpose(xyz))
    #uv = (int(fu + 0.5), int(fv + 0.5))
    uv = (fu, fv)
    print "Landmark image coords = ", uv
    x, y, z = xyz
    print "Range (slant) = ", math.sqrt(x*x + y*y + z*z)
    print "Range (ground) = ", math.sqrt(x*x + z*z)
    print "Azimuth = ", math.atan(x/z) * pu.RAD2DEG
    print

    print "Simulate different camera orientation:"
    print "Camera elevation = ", arb_ele
    print "Camera azimuth = ", arb_azi
    xyzr = rot_axes_rpy_deg(xyz, arb_ele, arb_azi, 0)
    print(xyzr)
    u, v = cam.project_xyz_to_uv(xyzr)
    print "Landmark NEW image coords = ", (u, v)
    print

    print "Using CameraHelper routine..."
    print

    print "Landmark is known to be at:"
    print xyz
    print "Landmark seen in image at = ", (u, v)
    print "Result:"
    xyz_1 = cam.calc_rel_xyz_to_landmark(xyz[1], u, v, arb_ele * pu.DEG2RAD)
    print xyz_1
    x, y, z = xyz_1
    print "Range (slant) = ", math.sqrt(x*x + y*y + z*z)
    print "Range (ground) = ", math.sqrt(x*x + z*z)
    print "Azimuth = ", math.atan(x/z) * pu.RAD2DEG
    print

    print "Second landmark known to be at:"
    xyz2 = np.float32([1.0, landmark_y, arb_z])
    print xyz2
    print "Rotate:"
    xyz2r = rot_axes_rpy_deg(xyz2, arb_ele, arb_azi, 0)
    print xyz2r
    print "Range to landmark #2 = ", np.linalg.norm(xyz2)
    u2, v2 = cam.project_xyz_to_uv(xyz2r)
    print "Landmark #2 image coords = ", (u2, v2)
    print "Result:"
    xyz_2 = cam.calc_rel_xyz_to_landmark(xyz2[1], u2, v2, arb_ele * pu.DEG2RAD)
    print xyz_2
    x, y, z = xyz_2
    print "Range (slant) = ", math.sqrt(x*x + y*y + z*z)
    print "Range (ground) = ", math.sqrt(x*x + z*z)
    print "Azimuth = ", math.atan(x/z) * pu.RAD2DEG
    print

    print "How do we triangulate?  I don't know yet..."
    print xyz_1
    print xyz_2
    xyz_3 = xyz_2 - xyz_1
    print xyz_3

    xz1 = np.array([xyz_1[0], xyz_1[2]])
    xz3 = np.array([xyz_3[0], xyz_3[2]])
    print xz1
    print xz3
    print np.inner(xz1, xz3)

    print "Done."
