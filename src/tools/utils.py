import numpy as np
import os
import cv2
import csv
import time
import math
from src.modules.OpenposeAPI import openpose_header
 

video_path = '../video'
image_path = '../image'

os.makedirs(image_path, exist_ok=True)


def save_video(frame_list, W, H, save_path):
    size = (int(W), int(H))
    encoder = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter()
    writer.open(save_path, encoder, fps=30, frameSize=size, isColor=True)

    for frame in frame_list:
        writer.write(frame)

    print(save_path + ' saved.')
    writer.release()


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
    pose_dict = {}
    for i, key in enumerate(openpose_header):
        pose_dict[key] = [pose_array[0, i, 0], pose_array[0, i, 1]]
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


def plot_skeleton_colorful(img, dict_data, thick=3, facial_feature=True):
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
    if joints_coor['nose'][0] != -1 and joints_coor['neck'][0] != -1:
        img = cv2.line(img, joints_coor['nose'], joints_coor['neck'], (66, 118, 228), thickness=thick)
    if joints_coor['left_hip'][0] != -1 and joints_coor['right_hip'][0] != -1:
        img = cv2.line(img, joints_coor['left_hip'], joints_coor['right_hip'], (146, 58, 128), thickness=thick)
    if joints_coor['neck'][0] != -1 and joints_coor['left_shoulder'][0] != -1:
        img = cv2.line(img, joints_coor['neck'], joints_coor['left_shoulder'], (66, 218, 128), thickness=thick)
    if joints_coor['neck'][0] != -1 and joints_coor['right_shoulder'][0] != -1:
        img = cv2.line(img, joints_coor['neck'], joints_coor['right_shoulder'], (250, 203, 91), thickness=thick)
    if joints_coor['right_shoulder'][0] != -1 and joints_coor['right_elbow'][0] != -1:
        img = cv2.line(img, joints_coor['right_shoulder'], joints_coor['right_elbow'], (35, 198, 77), thickness=thick)
    if joints_coor['right_elbow'][0] != -1 and joints_coor['right_wrist'][0] != -1:
        img = cv2.line(img, joints_coor['right_elbow'], joints_coor['right_wrist'], (35, 98, 177), thickness=thick)
    if joints_coor['left_shoulder'][0] != -1 and joints_coor['left_elbow'][0] != -1:
        img = cv2.line(img, joints_coor['left_shoulder'], joints_coor['left_elbow'], (62, 121, 58), thickness=thick)
    if joints_coor['left_elbow'][0] != -1 and joints_coor['left_wrist'][0] != -1:
        img = cv2.line(img, joints_coor['left_elbow'], joints_coor['left_wrist'], (23, 25, 118), thickness=thick)
    if joints_coor['right_hip'][0] != -1 and joints_coor['right_knee'][0] != -1:
        img = cv2.line(img, joints_coor['right_hip'], joints_coor['right_knee'], (94, 160, 66), thickness=thick)
    if joints_coor['right_knee'][0] != -1 and joints_coor['right_ankle'][0] != -1:
        img = cv2.line(img, joints_coor['right_knee'], joints_coor['right_ankle'], (44, 159, 96), thickness=thick)
    if joints_coor['neck'][0] != -1 and joints_coor['left_hip'][0] != -1:
        img = cv2.line(img, joints_coor['neck'], joints_coor['left_hip'], (152, 59, 98), thickness=thick)
    if joints_coor['neck'][0] != -1 and joints_coor['right_hip'][0] != -1:
        img = cv2.line(img, joints_coor['neck'], joints_coor['right_hip'], (51, 135, 239), thickness=thick)
    if joints_coor['left_hip'][0] != -1 and joints_coor['left_knee'][0] != -1:
        img = cv2.line(img, joints_coor['left_hip'], joints_coor['left_knee'], (75, 58, 217), thickness=thick)
    if joints_coor['left_knee'][0] != -1 and joints_coor['left_ankle'][0] != -1:
        img = cv2.line(img, joints_coor['left_knee'], joints_coor['left_ankle'], (244, 59, 166), thickness=thick)
    if facial_feature:
        if joints_coor['nose'][0] != -1 and joints_coor['right_eye'][0] != -1:
            img = cv2.line(img, joints_coor['nose'], joints_coor['right_eye'], (49, 56, 218), thickness=thick)
        if joints_coor['right_eye'][0] != -1 and joints_coor['right_ear'][0] != -1:
            img = cv2.line(img, joints_coor['right_eye'], joints_coor['right_ear'], (23, 25, 118), thickness=thick)
        if joints_coor['nose'][0] != -1 and joints_coor['left_eye'][0] != -1:
            img = cv2.line(img, joints_coor['nose'], joints_coor['left_eye'], (130, 35, 158), thickness=thick)
        if joints_coor['left_eye'][0] != -1 and joints_coor['left_ear'][0] != -1:
            img = cv2.line(img, joints_coor['left_eye'], joints_coor['left_ear'], (53, 200, 18), thickness=thick)
    for joint in dict_data.keys():
        if joint == 'None':
            continue
        if not facial_feature:
            if joint in ['right_eye', 'left_eye', 'right_ear', 'left_ear']:
                continue
        if joints_coor[joint] != (-1, -1):
            img = cv2.circle(img, joints_coor[joint], 5, (68, 147, 200), -1)
    return img


def plot_skeleton(img, dict_data, thick=3, color=(0, 255, 0), facial_feature=True):
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
    if joints_coor['nose'][0] != -1 and joints_coor['neck'][0] != -1:
        img = cv2.line(img, joints_coor['nose'], joints_coor['neck'], color, thickness=thick)
    if joints_coor['left_hip'][0] != -1 and joints_coor['right_hip'][0] != -1:
        img = cv2.line(img, joints_coor['left_hip'], joints_coor['right_hip'], color, thickness=thick)
    if joints_coor['neck'][0] != -1 and joints_coor['left_shoulder'][0] != -1:
        img = cv2.line(img, joints_coor['neck'], joints_coor['left_shoulder'], color, thickness=thick)
    if joints_coor['neck'][0] != -1 and joints_coor['right_shoulder'][0] != -1:
        img = cv2.line(img, joints_coor['neck'], joints_coor['right_shoulder'], color, thickness=thick)
    if joints_coor['right_shoulder'][0] != -1 and joints_coor['right_elbow'][0] != -1:
        img = cv2.line(img, joints_coor['right_shoulder'], joints_coor['right_elbow'], color, thickness=thick)
    if joints_coor['right_elbow'][0] != -1 and joints_coor['right_wrist'][0] != -1:
        img = cv2.line(img, joints_coor['right_elbow'], joints_coor['right_wrist'], color, thickness=thick)
    if joints_coor['left_shoulder'][0] != -1 and joints_coor['left_elbow'][0] != -1:
        img = cv2.line(img, joints_coor['left_shoulder'], joints_coor['left_elbow'], color, thickness=thick)
    if joints_coor['left_elbow'][0] != -1 and joints_coor['left_wrist'][0] != -1:
        img = cv2.line(img, joints_coor['left_elbow'], joints_coor['left_wrist'], color, thickness=thick)
    if joints_coor['right_hip'][0] != -1 and joints_coor['right_knee'][0] != -1:
        img = cv2.line(img, joints_coor['right_hip'], joints_coor['right_knee'], color, thickness=thick)
    if joints_coor['right_knee'][0] != -1 and joints_coor['right_ankle'][0] != -1:
        img = cv2.line(img, joints_coor['right_knee'], joints_coor['right_ankle'], color, thickness=thick)
    if joints_coor['neck'][0] != -1 and joints_coor['left_hip'][0] != -1:
        img = cv2.line(img, joints_coor['neck'], joints_coor['left_hip'], color, thickness=thick)
    if joints_coor['neck'][0] != -1 and joints_coor['right_hip'][0] != -1:
        img = cv2.line(img, joints_coor['neck'], joints_coor['right_hip'], color, thickness=thick)
    if joints_coor['left_hip'][0] != -1 and joints_coor['left_knee'][0] != -1:
        img = cv2.line(img, joints_coor['left_hip'], joints_coor['left_knee'], color, thickness=thick)
    if joints_coor['left_knee'][0] != -1 and joints_coor['left_ankle'][0] != -1:
        img = cv2.line(img, joints_coor['left_knee'], joints_coor['left_ankle'], color, thickness=thick)
    if facial_feature:
        if joints_coor['nose'][0] != -1 and joints_coor['right_eye'][0] != -1:
            img = cv2.line(img, joints_coor['nose'], joints_coor['right_eye'], color, thickness=thick)
        if joints_coor['right_eye'][0] != -1 and joints_coor['right_ear'][0] != -1:
            img = cv2.line(img, joints_coor['right_eye'], joints_coor['right_ear'], color, thickness=thick)
        if joints_coor['nose'][0] != -1 and joints_coor['left_eye'][0] != -1:
            img = cv2.line(img, joints_coor['nose'], joints_coor['left_eye'], color, thickness=thick)
        if joints_coor['left_eye'][0] != -1 and joints_coor['left_ear'][0] != -1:
            img = cv2.line(img, joints_coor['left_eye'], joints_coor['left_ear'], color, thickness=thick)
    for joint in dict_data.keys():
        if joint == 'None':
            continue
        if not facial_feature:
            if joint in ['right_eye', 'left_eye', 'right_ear', 'left_ear']:
                continue
        if joints_coor[joint] != (-1, -1):
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

