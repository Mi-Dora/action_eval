import numpy as np
import os
import cv2
import csv
import time
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dtaidistance import dtw_ndim, dtw, dtw_ndim_visualisation, dtw_visualisation
from src.tools.pose_angle import get_angle_vec, feature_len, plot_angle_curve
from src.modules.OpenposeAPI import openpose_header


def get_pose_angle_mask(angle_arr, percent=0.9):
    estimator = PCA()
    meanVals = np.mean(angle_arr, axis=0)
    meanRemoved = angle_arr - meanVals
    pca_angle_arr = estimator.fit_transform(angle_arr)
    plot_angle_curve(angle_arr, name='original')
    plot_angle_curve(pca_angle_arr, name='pca')
    ratio = estimator.explained_variance_ratio_
    mask = np.zeros_like(ratio)
    for i in range(ratio.size):
        if ratio[:i+1].sum() > percent:
            mask[:i+1] = 1
            break
    return pca_angle_arr, mask, estimator


def cal_warping_path(input_seq, template, mask=None):
    if mask is None:
        mask = np.ones(feature_len)
    distance, dtw_mat = dtw_ndim.warping_paths(input_seq*mask, template*mask)
    path = dtw.best_path(dtw_mat)

    # dtw_ndim_visualisation.plot_warping(input_seq, template, path, filename='plot_warping.png')

    # print(distance/len(path))
    return distance/len(path), path


def video_warping(input_video, template, path):
    new_video = []
    for step in path:
        input_frame = input_video[step[0]]
        template_frame = template[step[1]]
        new_frame = np.concatenate((input_frame, template_frame), axis=1)
        new_video.append(new_frame)
    return new_video


def pose_warping1(input_pose_arr, template_arr, path):
    new_input = []
    new_template = []
    for step in path:
        input_pose = input_pose_arr[step[0]]
        template_pose = template_arr[step[1]]
        new_input.append(input_pose)
        new_template.append(template_pose)
    return new_input, new_template


def pose_warping2(input_pose_arr, template_arr, path, mask):
    scores = []
    new_path = []
    new_path_scores = []
    for i, step in enumerate(path):
        input_pose = input_pose_arr[step[0]]
        template_pose = template_arr[step[1]]
        scores.append(angle_metrics(input_pose, template_pose, openpose_header, angle_weight=1.0, mask=mask))
        if len(new_path) > 0 and step[0] == new_path[-1][0]:
            if scores > new_path_scores[-1]:
                new_path_scores[-1] = scores
                new_path[-1][1] = step[1]
        else:
            new_path_scores.append(scores)
            new_path.append(step)
    return new_path, new_path_scores


def plot_path(paths, save_path):
    plt.figure(figsize=(5, 4))
    src = []
    template = []
    for i, path in enumerate(paths):
        path1 = np.array(path[0])
        path2 = np.array(path[1])
        path2[:, 0] += path1[-1, 0]
        path2[:, 1] += path1[-1, 1]
        path = np.concatenate((path1, path2), axis=0)
        for step in path:
            src.append(step[0])
            template.append((step[1]))
        # axis_max = max(path[-1][0], path[-1][1])
        # plt.xlim(0, axis_max)
        # plt.ylim(0, axis_max)
        plt.plot(path[:, 1], path[:, 0], label=str(i+1))

    # plt.plot(template, src, 'b^-')
    # plt.legend()
    lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.title('DTW Optimal Path')
    plt.xlabel('Template')
    plt.ylabel('Input Video')
    plt.savefig(save_path, bbox_extra_artists=(lg,), bbox_inches='tight')
    plt.clf()


def cal_seg_score(scores):
    key_frame_score = (scores[0] + scores[-1]) / 2
    other_score = scores[1:-1].sum()/(scores.size - 2)
    return (key_frame_score + other_score) / 2


def eval(pose_src, pose_ref, facial_feature=False, method='error', orient_weight=0.7, mask=None):
    """
    :param pose_src: (dict) pose
    :param pose_ref: (dict) pose
    :param facial_feature: (bool) whether facial features are considered in the metrics
    :return: (float 0~1) score
    """
    if mask is None:
        mask = np.ones(feature_len)
    # maybe convert dictionary to ndarray first
    pose_src_array = []
    pose_ref_array = []
    header = []
    for (key1, key2) in zip(pose_src, pose_ref):
        # pose1 and pos2 should have identical key sequence
        if not facial_feature:
            if key1 in ['right_eye', 'left_eye', 'right_ear', 'left_ear']:
                continue
        header.append(key1)
        pose_src_array.append(pose_src[key1])
        pose_ref_array.append(pose_ref[key2])
    pose_src_array = np.float64(pose_src_array)
    pose_ref_array = np.float64(pose_ref_array)
    if method == 'error':
        return error_metrics(pose_src_array, pose_ref_array, header, orient_weight)
    elif method == 'angle':
        return angle_metrics(pose_src_array, pose_ref_array, header, orient_weight, mask)


def error_metrics(pose_src_array, pose_ref_array, header, orient_weight=0.7):
    norm_position_err = get_position_error(pose_src_array, pose_ref_array, header)

    l_shoulder_idx = header.index('left_shoulder')
    r_shoulder_idx = header.index('right_shoulder')
    l_hip_idx = header.index('left_hip')
    r_hip_idx = header.index('right_hip')

    l_elbow_idx = header.index('left_elbow')
    r_elbow_idx = header.index('right_elbow')
    l_wrist_idx = header.index('left_wrist')
    r_wrist_idx = header.index('right_wrist')

    l_knee_idx = header.index('left_knee')
    r_knee_idx = header.index('right_knee')
    l_ankle_idx = header.index('left_ankle')
    r_ankle_idx = header.index('right_ankle')
    va, vb, vc, vd, ve, vf, vg, vh = [], [], [], [], [], [], [], []
    for pose in (pose_src_array, pose_ref_array):
        va.append(pose[l_shoulder_idx, :] - pose[l_elbow_idx, :])
        vb.append(pose[l_elbow_idx, :] - pose[l_wrist_idx, :])
        vc.append(pose[r_shoulder_idx, :] - pose[r_elbow_idx, :])
        vd.append(pose[r_elbow_idx, :] - pose[r_wrist_idx, :])
        ve.append(pose[l_hip_idx, :] - pose[l_knee_idx, :])
        vf.append(pose[l_knee_idx, :] - pose[l_ankle_idx, :])
        vg.append(pose[r_hip_idx, :] - pose[r_knee_idx, :])
        vh.append(pose[r_knee_idx, :] - pose[r_ankle_idx, :])
    vectors = np.float32([va, vb, vc, vd, ve, vf, vg, vh])
    """
    vector: (ndarray 8x2x2)
            [
                [ v1a[x, y], v2a[x, y] ], 
                [ v1b[x, y], v2b[x, y] ], 
                ...,
                [ v1h[x, y], v2h[x, y] ]
            ]
    """
    dot_prod = (vectors[:, 0, :] * vectors[:, 1, :]).sum(axis=1)
    norm = np.sqrt((vectors**2).sum(axis=2))
    vec_cos = dot_prod / norm[:, 0] / norm[:, 1]
    vec_theta = np.arccos(vec_cos).sum() / len(vec_cos)

    score = np.exp(-((1-orient_weight)*norm_position_err + orient_weight*vec_theta))
    return score


def angle_metrics(pose_src_array, pose_ref_array, header, angle_weight, mask):
    norm_position_err = get_position_error(pose_src_array, pose_ref_array, header)

    angle_vec_src = get_angle_vec(pose_src_array, header) * mask / np.pi
    angle_vec_ref = get_angle_vec(pose_ref_array, header) * mask / np.pi

    error_angle = np.sqrt(((angle_vec_src - angle_vec_ref)**2).sum())
    score = np.exp(-((1 - angle_weight) * norm_position_err + angle_weight * error_angle))
    return score


def get_position_error(pose_src_array, pose_ref_array, header):
    position_err = pose_src_array - pose_ref_array
    position_err = np.sqrt((position_err ** 2).sum(axis=1))
    l_shoulder_idx = header.index('left_shoulder')
    r_shoulder_idx = header.index('right_shoulder')
    l_hip_idx = header.index('left_hip')
    r_hip_idx = header.index('right_hip')
    cShoulder = (pose_ref_array[l_shoulder_idx, :] + pose_ref_array[r_shoulder_idx, :]) / 2
    cHip = (pose_ref_array[l_hip_idx, :] + pose_ref_array[r_hip_idx, :]) / 2
    meter = np.sqrt(((cShoulder - cHip) ** 2).sum())
    norm_position_err = (position_err / meter).sum() / len(position_err)
    return norm_position_err


if __name__ == '__main__':
    pass
