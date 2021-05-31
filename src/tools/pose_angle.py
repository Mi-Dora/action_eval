import numpy as np
import cv2
import os
import time
import json
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from src.tools.utils import openpose_header, plot_skeleton

vector_set = [
    ['left_elbow', 'left_wrist', 'left_shoulder', 'left_elbow'],
    ['right_shoulder', 'right_elbow', 'right_elbow', 'right_wrist'],
    ['left_shoulder', 'left_hip', 'left_shoulder', 'left_elbow'],
    ['right_shoulder', 'right_elbow', 'right_shoulder', 'right_hip'],
    ['left_hip', 'right_hip', 'left_hip', 'left_knee'],
    ['right_hip', 'right_knee', 'right_hip', 'left_hip'],
    ['left_knee', 'left_ankle', 'left_hip', 'left_knee'],
    ['right_hip', 'right_knee', 'right_knee', 'right_ankle'],
    ['neck', 'nose', 'left_shoulder', 'right_shoulder']
]
feature_len = len(vector_set)



def get_video_angle_vec(pose_array_list, action_class=None):
    """
    :param pose_array_list:
    :param action_class:
    :return: list of static frames
    """
    # Open the input movie file
    # angle method
    mask = np.ones(feature_len)
    if action_class is not None:
        f = open(os.path.split(os.path.realpath(__file__))[0] + '/../../config/action_class_mask.json', 'r')
        content = f.read()
        action_cfg = json.loads(content)
        mask = np.array(action_cfg[action_class])
        f.close()
    angle_vecs = []
    error_vecs = []
    frame_list = []
    for pose in pose_array_list:

        angle_vec = get_angle_vec(pose, openpose_header)

        # # get rid of abnormal value
        # if len(angle_vecs) != 0:
        #     errs = angle_vec - angle_vecs[-1]
        #     if len(error_vecs) != 0:
        #         for i, err in enumerate(errs):
        #             if abs(err) > abs(error_vecs[-1][i]) + np.pi/2:
        #                 angle_vec[i] = angle_vecs[-1][i] + error_vecs[-1][i]
        #                 errs[i] = error_vecs[-1][i]
        #     error_vecs.append(errs)

        # print(angle_vec)
        angle_vecs.append(angle_vec)

    angle_vecs = np.float32(angle_vecs)
    # angle_vecs = column_smooth(angle_vecs)
    return angle_vecs, mask


def get_angle_vec(pose_array, header):
    # b_pt1, e_pt1, b_pt2, e_pt2
    # anti-clockwise

    angle_vec = []

    for vector in vector_set:
        angle_vec.append(cal_angle(
            pose_array[header.index(vector[0]), 0:2],
            pose_array[header.index(vector[1]), 0:2],
            pose_array[header.index(vector[2]), 0:2],
            pose_array[header.index(vector[3]), 0:2]
        ))
    return np.float32(angle_vec)


def cal_angle(b_pt1, e_pt1, b_pt2, e_pt2):
    """
    :param b_pt1: (ndarray) begin point 1 [x, y]
    :param e_pt1: (ndarray) end point 1 [x, y]
    :param b_pt2: (ndarray) begin point 2 [x, y]
    :param e_pt2: (ndarray) end point 2 [x, y]
    :return: (float -pi~pi) angle from vector1 to vector2 (anticlockwise rad)
    """
    vec1 = e_pt1 - b_pt1
    vec2 = e_pt2 - b_pt2
    dot_prod = (vec1 * vec2).sum()
    norm1 = np.sqrt((vec1**2).sum())
    norm2 = np.sqrt((vec2**2).sum())
    vec_cos = dot_prod / norm1 / norm2
    if vec_cos > 1:
        vec_cos = 1.0
    elif vec_cos < -1:
        vec_cos = -1.0
    theta = np.arccos(vec_cos)
    cross_prod = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    # Since the image coordinate is left-hand,
    # so when the cross product is negative, the angle is anticlockwise
    if cross_prod < 0:
        return theta
    else:
        return -theta
        # if theta > np.pi / 2:
        #     return 2*np.pi - theta
        # else:
        #     return -theta


def column_smooth(arrays):
    smoothed = []
    for c in range(arrays.shape[1]):
        smoothed.append(savgol_filter(arrays[:, c], 7, 1, mode='nearest'))
    return np.float32(np.transpose(smoothed))


def plot_angle_curve(angle_vecs, static_frames_idx=None, save_path='tmp.png', name=''):
    plt.figure(figsize=(6, 4))
    plt.title(name)
    plt.plot(angle_vecs[:, 0], color='green', label='left elbow')
    plt.plot(angle_vecs[:, 1], color='red', label='right elbow')
    plt.plot(angle_vecs[:, 2], color='skyblue', label='left shoulder')
    plt.plot(angle_vecs[:, 3], color='blue', label='right shoulder')
    plt.plot(angle_vecs[:, 4], label='left hip')
    plt.plot(angle_vecs[:, 5], label='right hip')
    plt.plot(angle_vecs[:, 6], label='left knee')
    plt.plot(angle_vecs[:, 7], label='right knee')
    plt.plot(angle_vecs[:, 8], label='head')
    plt.legend(loc='upper left')  # 显示图例
    if static_frames_idx is not None:
        for idx in static_frames_idx:
            plt.vlines(idx, -np.pi/2, 3*np.pi/2, colors="r", linestyles="dashed")
    plt.xlabel('Frame ID')
    plt.ylabel('Angle (rad)')
    plt.show()
    # plt.savefig(save_path)
    plt.clf()


def plot_angle_curve_seg(angle_vecs, segments, save_path='tmp.png'):
    plt.title('Joint Angle Sequence')
    plt.plot(angle_vecs[:, 0], color='green', label='left elbow')
    plt.plot(angle_vecs[:, 1], color='red', label='right elbow')
    plt.plot(angle_vecs[:, 2], color='skyblue', label='left shoulder')
    plt.plot(angle_vecs[:, 3], color='blue', label='right shoulder')
    plt.plot(angle_vecs[:, 4], label='left hip')
    plt.plot(angle_vecs[:, 5], label='right hip')
    plt.plot(angle_vecs[:, 6], label='left knee')
    plt.plot(angle_vecs[:, 7], label='right knee')
    plt.plot(angle_vecs[:, 8], label='head')
    plt.legend(loc='upper left')  # 显示图例
    for segment in segments:
        plt.vlines(segment[0], -np.pi, np.pi, colors="r", linestyles="dashed")
        plt.vlines(segment[1], -np.pi, np.pi, colors="g", linestyles="dashed")
    plt.vlines(segments[-1][2], -np.pi, np.pi, colors="r", linestyles="dashed")
    plt.xlabel('Frame ID')
    plt.ylabel('Angle (rad)')
    plt.savefig(save_path)
    plt.clf()


if __name__ == '__main__':
    begin = time.time()
    # angle_vecs = get_static_frames('../sample/sample4.avi', '../sample/sample4.csv', DEBUG=False)
    # csv_folder = '../sample/repeat_pattern'
    # save_path = '../result/repeat_pattern'
    # for root, _, files in os.walk(csv_folder):
    #     for file in files:
    #         csv_path = os.path.join(root, file)
    #         save_name = os.path.join(save_path, file.split('.')[0] + '_smoothed.png')
    #         vis_angle_curve(csv_path, save_name, is_smooth=True)
    #         print(save_name)

    end = time.time()
    print("Time cost per frame:" + str(end - begin))




