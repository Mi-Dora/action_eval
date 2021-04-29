import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from src.utils import load_video_csv_anno, plot_skeleton
from src.frame_diff import moving_detect


openpose_header = [
    'nose', 'neck',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle',
    'right_ear', 'right_ear',
    'left_ear', 'left_ear'
]


def get_static_frames(pose_list, DEBUG=False, plot_save_path='tmp.png'):
    """
    :param video_file: video file name or camera id (e.g. 0)
    :return: list of static frames
    """
    # Open the input movie file


    # angle method

    cnt = 0
    angle_vecs = []
    error_vecs = []
    frame_list = []
    for pose in pose_list:
        pose_array = []

        header = []
        for key in pose:
            if key in ['right_eye', 'left_eye', 'right_ear', 'left_ear']:
                continue
            header.append(key)
            pose_array.append(pose[key])
        pose_array = np.float32(pose_array)
        angle_vec = get_angle_vec(pose_array, header)

        # get rid of abnormal value
        if len(angle_vecs) != 0:
            errs = angle_vec - angle_vecs[-1]
            if len(error_vecs) != 0:
                for i, err in enumerate(errs):
                    if abs(err) > abs(error_vecs[-1][i]) + np.pi/2:
                        angle_vec[i] = angle_vecs[-1][i] + error_vecs[-1][i]
                        errs[i] = error_vecs[-1][i]
            error_vecs.append(errs)

        # print(angle_vec)
        angle_vecs.append(angle_vec)

    angle_vecs = np.float32(angle_vecs)
    angle_vecs = column_smooth(angle_vecs)

    # static_frames_idx = motion_detect_process(angle_vecs)
    static_frames_idx = moving_detect(video_file)
    # static_frames_idx = [1, 2, 3, 4, 5]
    static_frames = []
    for idx in static_frames_idx:
        static_frames.append(frame_list[idx])
    # All done!

    plot_angle_curve(angle_vecs, static_frames_idx=static_frames_idx, save_path=plot_save_path)
    if DEBUG:
        return angle_vecs
    else:
        return angle_vecs, static_frames_idx, frame_list, pose_list


def motion_detect_process(angle_vecs):

    motion_thresh = 0.5
    valid_joint_thresh = 0.1
    vote_thresh = 0.1
    rows, _ = angle_vecs.shape

    error_vecs = []
    for r in range(2, rows):
        err_vec = angle_vecs[r, :] - angle_vecs[r - 2, :]
        error_vecs.append(err_vec)
    error_vecs = np.float32(error_vecs)
    avg_err = abs(error_vecs).sum()/error_vecs.size
    norm_err = error_vecs / avg_err
    mask = (abs(norm_err) > motion_thresh).astype('uint8')
    norm_err = norm_err * mask
    valid_joint = mask.sum(axis=0)/mask.shape[0] > valid_joint_thresh
    print("valid joint: ", valid_joint)
    binary_error_vecs1 = (norm_err > 0).astype('uint8')
    binary_error_vecs2 = (norm_err < 0).astype('uint8')
    binary_error_vecs = binary_error_vecs1 + binary_error_vecs2*2
    rows, _ = error_vecs.shape
    var_vecs = []
    for r in range(2, rows):
        var_vec = (binary_error_vecs[r - 2, :] + binary_error_vecs[r, :]) == 3  # similar to xor, 1+2=3
        var_vecs.append(var_vec)
    var_vecs = np.array(var_vecs) * valid_joint
    static_frames_vec = var_vecs.sum(axis=1) / valid_joint.sum() >= vote_thresh
    static_frames_idx = []
    lifetime = 0
    for idx, frame in enumerate(static_frames_vec):
        if frame:
            if lifetime == 0:
                static_frames_idx.append(idx)
                lifetime = 6
        if lifetime > 0:
            lifetime -= 1

    # static_frames_idx = [idx for idx, frame in enumerate(static_frame_vec) if frame]

    return static_frames_idx


def get_angle_vec(pose_array, header):
    # b_pt1, e_pt1, b_pt2, e_pt2
    # anti-clockwise
    vector_set = [
        ['left_elbow', 'left_wrist', 'left_shoulder', 'left_elbow'],
        ['right_shoulder', 'right_elbow', 'right_elbow', 'right_wrist'],
        ['left_shoulder', 'left_hip', 'left_shoulder', 'left_elbow'],
        ['right_shoulder', 'right_elbow', 'right_shoulder', 'right_hip'],
        ['left_hip', 'right_hip', 'left_hip', 'left_knee'],
        ['right_hip', 'right_knee', 'right_hip', 'left_hip'],
        ['left_knee', 'left_ankle', 'left_hip', 'left_knee'],
        ['right_hip', 'right_knee', 'right_knee', 'right_ankle']
        # ['left_shoulder', 'nose', 'left_shoulder', 'right_shoulder']
    ]

    angle_vec = []

    for vector in vector_set:
        angle_vec.append(cal_angle(
            pose_array[header.index(vector[0]), :],
            pose_array[header.index(vector[1]), :],
            pose_array[header.index(vector[2]), :],
            pose_array[header.index(vector[3]), :]
        ))
    center_shoulder = (pose_array[header.index('left_shoulder'), :] + pose_array[header.index('right_shoulder'), :])/2
    angle_vec.append(cal_angle(
        center_shoulder,
        pose_array[header.index('nose'), :],
        pose_array[header.index('left_shoulder'), :],
        pose_array[header.index('right_shoulder'), :]
    ))
    return np.float32(angle_vec)


def cal_angle(b_pt1, e_pt1, b_pt2, e_pt2):
    """
    :param b_pt1: (ndarray) begin point 1 [x, y]
    :param e_pt1: (ndarray) end point 1 [x, y]
    :param b_pt2: (ndarray) begin point 2 [x, y]
    :param e_pt2: (ndarray) end point 2 [x, y]
    :return: (float 0~2*pi) angle from vector1 to vector2 (anticlockwise rad)
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


def plot_angle_curve(angle_vecs, static_frames_idx=None, save_path='tmp.png'):
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
    if static_frames_idx is not None:
        for idx in static_frames_idx:
            plt.vlines(idx, -np.pi/2, 3*np.pi/2, colors="r", linestyles="dashed")
    plt.xlabel('Frame ID')
    plt.ylabel('Angle (rad)')
    plt.savefig(save_path)
    plt.clf()


def vis_angle_curve(csv_file, save_path, is_smooth=True):
    annos = load_video_csv_anno(csv_file)

    angle_vecs = []
    error_vecs = []

    for i, pose in enumerate(annos):
        pose_array = []
        header = []
        for key in pose:
            if key in ['right_eye', 'left_eye', 'right_ear', 'left_ear']:
                continue
            header.append(key)
            pose_array.append(pose[key])
        pose_array = np.float32(pose_array)
        angle_vec = get_angle_vec(pose_array, header)

        # # get rid of abnormal value
        # if len(angle_vecs) != 0:
        #     errs = angle_vec - angle_vecs[-1]
        #     if len(error_vecs) != 0:
        #         for i, err in enumerate(errs):
        #             if abs(err) > abs(error_vecs[-1][i]) + np.pi / 2:
        #                 angle_vec[i] = angle_vecs[-1][i] + error_vecs[-1][i]
        #                 errs[i] = error_vecs[-1][i]
        #     error_vecs.append(errs)

        angle_vecs.append(angle_vec)
    angle_vecs = np.float32(angle_vecs)
    if is_smooth:
        angle_vecs = column_smooth(angle_vecs)
    plot_angle_curve(angle_vecs, save_path=save_path)


if __name__ == '__main__':
    begin = time.time()
    # angle_vecs = get_static_frames('../sample/sample4.avi', '../sample/sample4.csv', DEBUG=False)
    csv_folder = '../sample/repeat_pattern'
    save_path = '../result/repeat_pattern'
    for root, _, files in os.walk(csv_folder):
        for file in files:
            csv_path = os.path.join(root, file)
            save_name = os.path.join(save_path, file.split('.')[0] + '_smoothed.png')
            vis_angle_curve(csv_path, save_name, is_smooth=True)
            print(save_name)

    end = time.time()
    print("Time cost per frame:" + str(end - begin))




