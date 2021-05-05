import numpy as np
import os
import cv2
import csv
import time
import math


def get_transform_matrix(pose_src, pose_ref, method='SoftRigid'):
    """
    :param pose_src: (dict) source pose
    :param pose_ref: (dict) reference pose
    :param method: (str) transform method (offer 'Affine' or 'SoftRigid')
    :return: [2x3] matrix for 'Affine', two [2x3] matrices for SoftRigid
    """
    if method == 'Affine':
        return AffineTransform(pose_src, pose_ref)
    elif method == 'SoftRigid':
        return SoftRigidTransform(pose_src, pose_ref)


def AffineTransform(pose_src, pose_ref):
    """
    :param pose_src: (dict) source pose
    :param pose_ref: (dict) reference pose
    :return:
    """
    lHip_src = pose_src['left_hip']
    rHip_src = pose_src['right_hip']
    cHip_src = ((lHip_src[0]+rHip_src[0])/2, (lHip_src[1]+rHip_src[1])/2)
    lShoulder_src = pose_src['left_shoulder']
    rShoulder_src = pose_src['right_shoulder']

    lHip_ref = pose_ref['left_hip']
    rHip_ref = pose_ref['right_hip']
    cHip_ref = ((lHip_ref[0]+rHip_ref[0])/2, (lHip_ref[1]+rHip_ref[1])/2)
    lShoulder_ref = pose_ref['left_shoulder']
    rShoulder_ref = pose_ref['right_shoulder']

    src = np.float32((cHip_src, lShoulder_src, rShoulder_src))
    ref = np.float32((cHip_ref, lShoulder_ref, rShoulder_ref))
    mat = cv2.getAffineTransform(src, ref)
    return mat


def SoftRigidTransform(pose_src, pose_ref):
    lHip_src = pose_src['left_hip']
    rHip_src = pose_src['right_hip']
    (x1, y1) = ((lHip_src[0]+rHip_src[0])/2, (lHip_src[1]+rHip_src[1])/2)
    lShoulder_src = pose_src['left_shoulder']
    rShoulder_src = pose_src['right_shoulder']
    (x2, y2) = ((lShoulder_src[0]+rShoulder_src[0])/2, (lShoulder_src[1]+rShoulder_src[1])/2)

    lHip_ref = pose_ref['left_hip']
    rHip_ref = pose_ref['right_hip']
    (x1_, y1_) = ((lHip_ref[0]+rHip_ref[0])/2, (lHip_ref[1]+rHip_ref[1])/2)
    lShoulder_ref = pose_ref['left_shoulder']
    rShoulder_ref = pose_ref['right_shoulder']
    (x2_, y2_) = ((lShoulder_ref[0]+rShoulder_ref[0])/2, (lShoulder_ref[1]+rShoulder_ref[1])/2)
    dx = x1 - x2
    dy = y1 - y2
    dx_ = x1_ - x2_
    dy_ = y1_ - y2_
    a = np.sqrt((dx_**2+dy_**2)/(dx**2+dy**2))
    sin_theta1 = (-dx_*dy + dx*dy_) / (a*(dx**2 + dy**2))
    sin_theta2 = (-dx_*dy - dx*dy_) / (a*(dx**2 + dy**2))
    cos_theta1 = (dx*dx_ + dy*dy_) / (a*(dx**2 + dy**2))
    cos_theta2 = (dx*dx_ - dy*dy_) / (a*(dx**2 + dy**2))
    theta1 = math.atan2(cos_theta1, sin_theta1) / np.pi * 180
    theta2 = math.atan2(cos_theta2, sin_theta2) / np.pi * 180
    px1 = x1_ - a * cos_theta1 * x1 + a * sin_theta1 * y1
    py1 = y1_ - a * sin_theta1 * x1 - a * cos_theta1 * y1
    px2 = x1_ - a * cos_theta2 * x1 + a * sin_theta2 * y1
    py2 = y1_ - a * sin_theta2 * x1 - a * cos_theta2 * y1
    # The experiment shows that mat1 should be right
    mat1 = np.float64(
        [
            [a*cos_theta1, -a*sin_theta1, px1],
            [a*sin_theta1,  a*cos_theta1, py1],
            [0, 0, 1]
        ]
    )
    mat2 = np.float64(
        [
            [a * cos_theta2, -a * sin_theta2, px2],
            [a * sin_theta2, a * cos_theta2, py2],
            [0, 0, 1]
        ]
    )
    return mat1, mat2


def transform_pose(mat, src_pose):
    """
    :param mat: (ndarray 2x3 or 3x3)
    :param src_pose: (dict) source pose
    :return: dst_pose: (dict) destination pose
    """
    dst_pose = {}
    for key in src_pose.keys():
        src_coor = np.float64([[src_pose[key][0]], [src_pose[key][1]], [1]])
        dst_coor = mat.dot(src_coor)
        dst_pose[key] = (dst_coor[0, 0], dst_coor[1, 0])
    return dst_pose


def trans_test(x1, y1, x2, y2, x1_, y1_, x2_, y2_):
    dx = x1 - x2
    dy = y1 - y2
    dx_ = x1_ - x2_
    dy_ = y1_ - y2_
    a = np.sqrt((dx_ ** 2 + dy_ ** 2) / (dx ** 2 + dy ** 2))
    sin_theta1 = (-dx_ * dy + dx * dy_) / (a * (dx ** 2 + dy ** 2))
    sin_theta2 = (-dx_ * dy - dx * dy_) / (a * (dx ** 2 + dy ** 2))
    cos_theta1 = (dx * dx_ + dy * dy_) / (a * (dx ** 2 + dy ** 2))
    cos_theta2 = (dx * dx_ - dy * dy_) / (a * (dx ** 2 + dy ** 2))
    print('a = %f' % a)

    if -1.0 <= sin_theta1 <= 1.0:
        theta1 = np.arcsin(sin_theta1) / np.pi * 180
        print("theta1 = %f" % theta1)
        print('sin_theta1 = %f' % sin_theta1)
    else:
        print('sin_theta1 = %f' % sin_theta1)
    if -1.0 <= sin_theta2 <= 1.0:
        theta2 = np.arcsin(sin_theta2) / np.pi * 180
        print("theta2 = %f" % theta2)
        print('sin_theta2 = %f' % sin_theta2)
    else:
        print('sin_theta2 = %f' % sin_theta2)

    if -1.0 <= cos_theta1 <= 1.0:
        theta1 = np.arccos(cos_theta1) / np.pi * 180
        print("\ntheta1 = %f" % theta1)
        print('cos_theta1 = %f' % cos_theta1)
    else:
        print('cos_theta1 = %f' % cos_theta1)
    if -1.0 <= cos_theta2 <= 1.0:
        theta2 = np.arccos(cos_theta2) / np.pi * 180
        print("theta2 = %f" % theta2)
        print('cos_theta2 = %f' % cos_theta2)
    else:
        print('cos_theta2 = %f' % cos_theta2)
    theta1 = math.atan2(cos_theta1, sin_theta1) / np.pi * 180
    theta2 = math.atan2(cos_theta2, sin_theta2) / np.pi * 180
    print('\n')
    print("theta1 = %f" % theta1)
    print("theta2 = %f" % theta2)


if __name__ == '__main__':
    trans_test(1, 0, 0, 1, 0, 0, 1, 2)
