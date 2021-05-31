import numpy as np
import os
import cv2
import csv
import time
import math
import matplotlib.pyplot as plt
from src.modules.OpenposeAPI import openpose_header
from scipy.signal import savgol_filter
from src.tools. exponential_smooth import double_exponential_smoothing
 

# video_path = '../video'
image_path = '../image'
#
# os.makedirs(image_path, exist_ok=True)
# os.makedirs(video_path, exist_ok=True)


def plot_circle(ax, x_c, y_c, r):
    theta = np.linspace(0, 2 * np.pi, 200)
    x = r * np.cos(theta) + x_c
    y = r * np.sin(theta) + y_c
    ax.plot(x, y, color="green", linewidth=1)


def plot_pose_traj(pose_arr, pose_arr_smooth, action_name='', part=''):
    traj1 = pose_arr[:, openpose_header.index('left_'+part), 0:2]
    traj2 = pose_arr[:, openpose_header.index('right_'+part), 0:2]
    traj = np.concatenate((traj1, traj2), axis=0)
    traj[:, 0] = (traj[:, 0] - traj[:, 0].min()) / (traj[:, 0].max() - traj[:, 0].min())
    traj[:, 1] = (traj[:, 1] - traj[:, 1].min()) / (traj[:, 1].max() - traj[:, 1].min())
    traj1_s = pose_arr_smooth[:, openpose_header.index('left_'+part), 0:2]
    traj2_s = pose_arr_smooth[:, openpose_header.index('right_'+part), 0:2]
    traj_s = np.concatenate((traj1_s, traj2_s), axis=0)
    traj_s[:, 0] = (traj_s[:, 0] - traj_s[:, 0].min()) / (traj_s[:, 0].max() - traj_s[:, 0].min())
    traj_s[:, 1] = (traj_s[:, 1] - traj_s[:, 1].min()) / (traj_s[:, 1].max() - traj_s[:, 1].min())
    plt.figure(figsize=(6, 4))
    plt.scatter(traj_s[:, 0], traj_s[:, 1], s=1, label=part+' Locus')
    plt.scatter(traj[:, 0], traj[:, 1], s=1, label=part+' Locus w/o patch+smooth')
    plt.legend()
    plt.title('Key-point Locus: '+action_name)
    # plt.savefig('../../plots/traj/.png')
    plt.show()
    plt.clf()


def pose_smooth(pose_arr):
    x_mat = pose_arr[:, :, 0]
    y_mat = pose_arr[:, :, 1]
    for c in range(pose_arr.shape[1]):
        x_mat[:, c] = double_exponential_smoothing(x_mat[:, c], alpha=0.3, beta=0.3)
        y_mat[:, c] = double_exponential_smoothing(y_mat[:, c], alpha=0.3, beta=0.3)
    return pose_arr


def pose_smooth1(pose_arr):
    # smoothed = []
    x_mat = pose_arr[:, :, 0]
    y_mat = pose_arr[:, :, 1]
    for c in range(pose_arr.shape[1]):
        x_mat[:, c] = savgol_filter(x_mat[:, c], 7, 1, mode='nearest')
        y_mat[:, c] = savgol_filter(y_mat[:, c], 7, 1, mode='nearest')
    return pose_arr


def pose_patch(pose_arr):
    confidence_map = pose_arr[:, :, 2]
    mask = confidence_map == 0
    count_map = np.zeros_like(mask, dtype=int)
    begin_map = np.zeros_like(pose_arr[:, :, 0:2])  # (begin_x, begin_y)
    begin_idx_map = np.zeros_like(mask, dtype=int)
    end_map = np.zeros_like(pose_arr[:, :, 0:2])  # (end_x, end_y)
    end_idx_map = np.zeros_like(mask, dtype=int)
    counter = np.zeros_like(mask[0], dtype=int)
    num_frame = pose_arr.shape[0]
    for idx in range(num_frame):
        counter += mask[idx]
        counter[mask[idx, :] == 0] = 0
        begin_map[idx][mask[idx, :] == 0] = pose_arr[idx, :, 0:2][mask[idx, :] == 0]
        begin_idx_map[idx][mask[idx, :] == 0] = idx
        if idx > 0:
            begin_map[idx][mask[idx, :] == 1] = begin_map[idx - 1][mask[idx, :] == 1]
            begin_idx_map[idx][mask[idx, :] == 1] = begin_idx_map[idx - 1][mask[idx, :] == 1]
        count_map[idx] = counter
    for idx in range(num_frame-1, -1, -1):
        end_map[idx][mask[idx, :] == 0] = pose_arr[idx, :, 0:2][mask[idx, :] == 0]
        end_idx_map[idx][mask[idx, :] == 0] = idx
        if idx < num_frame - 1:
            end_map[idx][mask[idx, :] == 1] = end_map[idx + 1][mask[idx, :] == 1]
            end_idx_map[idx][mask[idx, :] == 1] = end_idx_map[idx + 1][mask[idx, :] == 1]
    begin_map[begin_idx_map == 0] = end_map[begin_idx_map == 0]
    begin_idx_map[begin_idx_map == 0] = end_idx_map[begin_idx_map == 0]
    end_map[end_idx_map == 0] = begin_map[end_idx_map == 0]
    end_idx_map[end_idx_map == 0] = begin_idx_map[end_idx_map == 0]

    gap_map = end_idx_map - begin_idx_map
    gap_map[gap_map == 0] = 1

    pose_arr[:, :, 0] += mask * (begin_map[:, :, 0] + count_map * (end_map[:, :, 0] - begin_map[:, :, 0]) / gap_map)
    pose_arr[:, :, 1] += mask * (begin_map[:, :, 1] + count_map * (end_map[:, :, 1] - begin_map[:, :, 1]) / gap_map)
    return pose_arr


def plot_score_curve(score_list, segments, save_path):
    plt.ylim(0, 1)
    score_arr = np.array(score_list)
    score_arr = savgol_filter(score_arr, 7, 1, mode='nearest')
    plt.plot(score_arr)
    for segment in segments:
        plt.vlines(segment[0], 0, 1, colors="r", linestyles="dashed")
        plt.vlines(segment[1], 0, 1, colors="g", linestyles="dashed")
    plt.vlines(segments[-1][2], 0, 1, colors="r", linestyles="dashed")
    plt.title('Time-Similarity Curve')
    plt.xlabel('Time Line')
    plt.ylabel('Similarity')
    plt.savefig(save_path)
    plt.clf()


def auto_text(rects, plt):
    for rect in rects:
        plt.text(rect.get_x() + rect.get_width()/2, rect.get_height(), '%.1f' % rect.get_height(), ha='center', va='bottom')


def plot_bar(seg_scores, save_path):
    colors = []
    for i, score in enumerate(seg_scores):
        if score >= 0.9:
            colors.append('#4eaf4e')
        elif 0.7 < score < 0.9:
            colors.append('#ff7f0e')
        else:
            colors.append('#dd4a4a')
        seg_scores[i] *= 100
    plt.figure()
    # plt.ylim(0, 1)
    rects = plt.bar(range(1, len(seg_scores)+1), seg_scores, color=colors, edgecolor='darkblue')  # '#9999ff'
    plt.title('Period-Similarity Bar')
    plt.xlabel('Period')
    plt.ylabel('Similarity')
    auto_text(rects, plt)
    plt.savefig(save_path)
    plt.clf()


def save_video(frame_list, save_path):
    H, W, _ = frame_list[0].shape
    size = (int(W), int(H))
    encoder = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter()
    writer.open(save_path, encoder, fps=30, frameSize=size, isColor=True)

    for frame in frame_list:
        writer.write(frame)

    print(save_path + ' saved.')
    writer.release()


def init_video_writer(H, W, save_path):
    size = (int(W), int(H))
    encoder = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter()
    writer.open(save_path, encoder, fps=30, frameSize=size, isColor=True)
    return writer


def video2image(video_file):
    os.makedirs(os.path.join(image_path, os.path.basename(video_file).split('.')[0]), exist_ok=True)
    # Open the input movie file
    input_movie = cv2.VideoCapture(video_file)
    cnt = 0
    while True:
        ret, frame = input_movie.read()
        if not ret:
            break
        f_name = os.path.join(image_path, os.path.basename(video_file).split('.')[0], '%06d.png' % cnt)
        cv2.imwrite(f_name, frame)
        cnt += 1

    # All done!
    print(video_file + ' saved.')
    input_movie.release()


def get_prior_person(data, is_hrnet=True):
    if is_hrnet:
        prior_data = []
        for i in range(0, 34):
            prior_data.append(int(data[i]))
        return prior_data
    return None


def array2dict(pose_array):
    # for single person pose
    pose_array_copy = pose_array.copy()
    pose_array_copy = np.squeeze(pose_array_copy)
    pose_dict = {}
    for i, key in enumerate(openpose_header):
        pose_dict[key] = [pose_array_copy[i, 0], pose_array_copy[i, 1]]
    return pose_dict


def _list2dict(data, header):
    dict_data = {}
    key = ''
    x = 0
    y = 0
    for i, column in enumerate(header):
        name = column[:-2]
        coor = column[-1]
        if coor == 'x':
            key = name
            x = data[i]
        if coor == 'y' and key == name:
            y = data[i]
            dict_data[key] = (x, y)
    return dict_data


def plot_skeleton_colorful(img, dict_data, thick=3, facial_feature=False):
    '''
        Args:
            img:            (PILImage) image for annotating
            dict_data:      (diopenpose_ct) keypoints saved in dictionary with keys in keypoint name
            thick:          (int) thick of the line
            facial_feature: (bool) whether to draw facial feature
    '''
    joints_coor = dict_data
    for key in joints_coor.keys():
        joints_coor[key] = (int(joints_coor[key][0]), int(joints_coor[key][1]))
    if joints_coor['nose'][0] != 0 and joints_coor['neck'][0] != 0:
        img = cv2.line(img, joints_coor['nose'], joints_coor['neck'], (66, 118, 228), thickness=thick)
    if joints_coor['left_hip'][0] != 0 and joints_coor['right_hip'][0] != 0:
        img = cv2.line(img, joints_coor['left_hip'], joints_coor['right_hip'], (146, 58, 128), thickness=thick)
    if joints_coor['neck'][0] != 0 and joints_coor['left_shoulder'][0] != 0:
        img = cv2.line(img, joints_coor['neck'], joints_coor['left_shoulder'], (66, 218, 128), thickness=thick)
    if joints_coor['neck'][0] != 0 and joints_coor['right_shoulder'][0] != 0:
        img = cv2.line(img, joints_coor['neck'], joints_coor['right_shoulder'], (250, 203, 91), thickness=thick)
    if joints_coor['right_shoulder'][0] != 0 and joints_coor['right_elbow'][0] != 0:
        img = cv2.line(img, joints_coor['right_shoulder'], joints_coor['right_elbow'], (35, 198, 77), thickness=thick)
    if joints_coor['right_elbow'][0] != 0 and joints_coor['right_wrist'][0] != 0:
        img = cv2.line(img, joints_coor['right_elbow'], joints_coor['right_wrist'], (35, 98, 177), thickness=thick)
    if joints_coor['left_shoulder'][0] != 0 and joints_coor['left_elbow'][0] != 0:
        img = cv2.line(img, joints_coor['left_shoulder'], joints_coor['left_elbow'], (62, 121, 58), thickness=thick)
    if joints_coor['left_elbow'][0] != 0 and joints_coor['left_wrist'][0] != 0:
        img = cv2.line(img, joints_coor['left_elbow'], joints_coor['left_wrist'], (23, 25, 118), thickness=thick)
    if joints_coor['right_hip'][0] != 0 and joints_coor['right_knee'][0] != 0:
        img = cv2.line(img, joints_coor['right_hip'], joints_coor['right_knee'], (94, 160, 66), thickness=thick)
    if joints_coor['right_knee'][0] != 0 and joints_coor['right_ankle'][0] != 0:
        img = cv2.line(img, joints_coor['right_knee'], joints_coor['right_ankle'], (44, 159, 96), thickness=thick)
    if joints_coor['neck'][0] != 0 and joints_coor['left_hip'][0] != 0:
        img = cv2.line(img, joints_coor['neck'], joints_coor['left_hip'], (152, 59, 98), thickness=thick)
    if joints_coor['neck'][0] != 0 and joints_coor['right_hip'][0] != 0:
        img = cv2.line(img, joints_coor['neck'], joints_coor['right_hip'], (51, 135, 239), thickness=thick)
    if joints_coor['left_hip'][0] != 0 and joints_coor['left_knee'][0] != 0:
        img = cv2.line(img, joints_coor['left_hip'], joints_coor['left_knee'], (75, 58, 217), thickness=thick)
    if joints_coor['left_knee'][0] != 0 and joints_coor['left_ankle'][0] != 0:
        img = cv2.line(img, joints_coor['left_knee'], joints_coor['left_ankle'], (244, 59, 166), thickness=thick)
    if facial_feature:
        if joints_coor['nose'][0] != 0 and joints_coor['right_eye'][0] != 0:
            img = cv2.line(img, joints_coor['nose'], joints_coor['right_eye'], (49, 56, 218), thickness=thick)
        if joints_coor['right_eye'][0] != 0 and joints_coor['right_ear'][0] != 0:
            img = cv2.line(img, joints_coor['right_eye'], joints_coor['right_ear'], (23, 25, 118), thickness=thick)
        if joints_coor['nose'][0] != 0 and joints_coor['left_eye'][0] != 0:
            img = cv2.line(img, joints_coor['nose'], joints_coor['left_eye'], (130, 35, 158), thickness=thick)
        if joints_coor['left_eye'][0] != 0 and joints_coor['left_ear'][0] != 0:
            img = cv2.line(img, joints_coor['left_eye'], joints_coor['left_ear'], (53, 200, 18), thickness=thick)
    for joint in dict_data.keys():
        if joint == 'None':
            continue
        if not facial_feature:
            if joint in ['right_eye', 'left_eye', 'right_ear', 'left_ear']:
                continue
        if joints_coor[joint] != (0, 0):
            img = cv2.circle(img, joints_coor[joint], 5, (68, 147, 200), -1)
    return img


def plot_skeleton(img, dict_data, thick=3, color=(0, 255, 0), facial_feature=False):
    '''
        Args:
            img:            (PILImage) image for annotating
            dict_data:      (diopenpose_ct) keypoints saved in dictionary with keys in keypoint name
            thick:          (int) thick of the line
            color:          (tuple) color for the skeleton
            facial_feature: (bool) whether to draw facial feature
    '''
    joints_coor = dict_data
    for key in joints_coor.keys():
        joints_coor[key] = (int(joints_coor[key][0]), int(joints_coor[key][1]))
    if joints_coor['nose'][0] != 0 and joints_coor['neck'][0] != 0:
        img = cv2.line(img, joints_coor['nose'], joints_coor['neck'], color, thickness=thick)
    if joints_coor['left_hip'][0] != 0 and joints_coor['right_hip'][0] != 0:
        img = cv2.line(img, joints_coor['left_hip'], joints_coor['right_hip'], color, thickness=thick)
    if joints_coor['neck'][0] != 0 and joints_coor['left_shoulder'][0] != 0:
        img = cv2.line(img, joints_coor['neck'], joints_coor['left_shoulder'], color, thickness=thick)
    if joints_coor['neck'][0] != 0 and joints_coor['right_shoulder'][0] != 0:
        img = cv2.line(img, joints_coor['neck'], joints_coor['right_shoulder'], color, thickness=thick)
    if joints_coor['right_shoulder'][0] != 0 and joints_coor['right_elbow'][0] != 0:
        img = cv2.line(img, joints_coor['right_shoulder'], joints_coor['right_elbow'], color, thickness=thick)
    if joints_coor['right_elbow'][0] != 0 and joints_coor['right_wrist'][0] != 0:
        img = cv2.line(img, joints_coor['right_elbow'], joints_coor['right_wrist'], color, thickness=thick)
    if joints_coor['left_shoulder'][0] != 0 and joints_coor['left_elbow'][0] != 0:
        img = cv2.line(img, joints_coor['left_shoulder'], joints_coor['left_elbow'], color, thickness=thick)
    if joints_coor['left_elbow'][0] != 0 and joints_coor['left_wrist'][0] != 0:
        img = cv2.line(img, joints_coor['left_elbow'], joints_coor['left_wrist'], color, thickness=thick)
    if joints_coor['right_hip'][0] != 0 and joints_coor['right_knee'][0] != 0:
        img = cv2.line(img, joints_coor['right_hip'], joints_coor['right_knee'], color, thickness=thick)
    if joints_coor['right_knee'][0] != 0 and joints_coor['right_ankle'][0] != 0:
        img = cv2.line(img, joints_coor['right_knee'], joints_coor['right_ankle'], color, thickness=thick)
    if joints_coor['neck'][0] != 0 and joints_coor['left_hip'][0] != 0:
        img = cv2.line(img, joints_coor['neck'], joints_coor['left_hip'], color, thickness=thick)
    if joints_coor['neck'][0] != 0 and joints_coor['right_hip'][0] != 0:
        img = cv2.line(img, joints_coor['neck'], joints_coor['right_hip'], color, thickness=thick)
    if joints_coor['left_hip'][0] != 0 and joints_coor['left_knee'][0] != 0:
        img = cv2.line(img, joints_coor['left_hip'], joints_coor['left_knee'], color, thickness=thick)
    if joints_coor['left_knee'][0] != 0 and joints_coor['left_ankle'][0] != 0:
        img = cv2.line(img, joints_coor['left_knee'], joints_coor['left_ankle'], color, thickness=thick)
    if facial_feature:
        if joints_coor['nose'][0] != 0 and joints_coor['right_eye'][0] != 0:
            img = cv2.line(img, joints_coor['nose'], joints_coor['right_eye'], color, thickness=thick)
        if joints_coor['right_eye'][0] != 0 and joints_coor['right_ear'][0] != 0:
            img = cv2.line(img, joints_coor['right_eye'], joints_coor['right_ear'], color, thickness=thick)
        if joints_coor['nose'][0] != 0 and joints_coor['left_eye'][0] != 0:
            img = cv2.line(img, joints_coor['nose'], joints_coor['left_eye'], color, thickness=thick)
        if joints_coor['left_eye'][0] != 0 and joints_coor['left_ear'][0] != 0:
            img = cv2.line(img, joints_coor['left_eye'], joints_coor['left_ear'], color, thickness=thick)
    for joint in dict_data.keys():
        if joint == 'None':
            continue
        if not facial_feature:
            if joint in ['right_eye', 'left_eye', 'right_ear', 'left_ear']:
                continue
        if joints_coor[joint] != (0, 0):
            img = cv2.circle(img, joints_coor[joint], 5, (68, 147, 200), -1)
    return img


if __name__ == '__main__':
    pass
    # for root, _, files in os.walk(video_path):
    #     for file in files:
    #         video2image(os.path.join(root, file))
    # for root, _, files in os.walk(anno_path):
    #     for file in files:
    #         save_anno_per_img(os.path.join(root, file))
    # img = cv2.imread('../sample/sample.png')
    # src_pose = load_img_csv_anno('../sample/sample.csv')
    # ref_pose = load_img_csv_anno('../sample/reference.csv')

    # img = plot_skeleton(img, dict_data)
    # cv2.imwrite('../test.png', img)

